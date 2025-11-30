import os
import argparse
import itertools
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import bitsandbytes as bnb
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with Dice loss.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results_dreambooth_dice")
    parser.add_argument("--seg_model_weights", type=str, required=True)
    parser.add_argument("--unique_token", type=str, default="<nih-xray>")
    parser.add_argument("--class_token", type=str, default="x-ray")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray")
    parser.add_argument("--num_class_images", type=int, default=200)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dice_loss_weight", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        
        self.instance_images_path = sorted([os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # Robust matching: look for mask with same filename in mask dir
        self.instance_masks_path = [os.path.join(instance_mask_root, os.path.basename(f)) for f in self.instance_images_path]
        self.num_instance_images = len(self.instance_images_path)
        
        self.class_images_path = sorted([os.path.join(class_data_root, f) for f in os.listdir(class_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images, self.num_class_images)
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size), 
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        
        instance_mask = Image.open(self.instance_masks_path[index % self.num_instance_images]).convert("L")
        example["instance_masks"] = self.mask_transforms(instance_mask)
        
        if self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images]).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(self.class_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        return example

def collate_fn(examples, pad_token_id, with_prior_preservation=True):
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]
    masks = [e["instance_masks"] for e in examples]
    
    if with_prior_preservation:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]
        
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    masks = torch.stack(masks)
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in input_ids], batch_first=True, padding_value=pad_token_id)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "masks": masks}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.class_data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # --- Segmentation Model ---
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.eval()
    for param in seg_model.parameters(): param.requires_grad = False
    dice_loss_fn = DiceLoss(sigmoid=True)

    # --- Diffusion Setup ---
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    
    if tokenizer.add_tokens(args.unique_token) > 0:
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[tokenizer.convert_tokens_to_ids(args.unique_token)] = token_embeds[tokenizer.convert_tokens_to_ids(args.class_token)].clone()
    
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    unet.enable_gradient_checkpointing()
    try: unet.enable_xformers_memory_efficient_attention()
    except: pass
        
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing() # Saves VRAM during decode
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # --- Class Images ---
    if len(os.listdir(args.class_data_dir)) < args.num_class_images:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
        pipe.set_progress_bar_config(disable=True)
        for i in tqdm(range(args.num_class_images - len(os.listdir(args.class_data_dir)))):
            pipe(f"a photo of a {args.class_token}", num_inference_steps=50).images[0].save(os.path.join(args.class_data_dir, f"{i}.jpg"))
        del pipe
        torch.cuda.empty_cache()

    # --- Training Setup ---
    dataset = DreamBoothDataset(args.instance_data_dir, args.instance_mask_dir, args.class_data_dir, tokenizer, f"a photo of {args.unique_token}", f"a photo of a {args.class_token}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
    
    optimizer = bnb.optim.AdamW8bit(itertools.chain(unet.parameters(), text_encoder.parameters()), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(dataloader) * args.num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    # --- Resume Logic ---
    start_epoch = 0
    global_step = 0
    if os.path.exists(args.output_dir):
        checkpoints = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1]))
        if checkpoints and os.path.exists(os.path.join(args.output_dir, checkpoints[-1], "training_state.pth")):
            path = os.path.join(args.output_dir, checkpoints[-1], "training_state.pth")
            state = torch.load(path, map_location=device)
            unet.load_state_dict(state['unet']); text_encoder.load_state_dict(state['text_encoder'])
            optimizer.load_state_dict(state['optimizer']); lr_scheduler.load_state_dict(state['lr_scheduler'])
            scaler.load_state_dict(state['scaler']); start_epoch = state['epoch']; global_step = state['global_step']
            print(f"Resumed from epoch {start_epoch}")

    # --- Loop ---
    for epoch in range(start_epoch, args.num_epochs):
        unet.train(); text_encoder.train()
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.amp.autocast('cuda'):
                noise_pred = unet(noisy_latents, timesteps, text_encoder(batch["input_ids"].to(device))[0]).sample
                chunk_pred, chunk_noise = noise_pred.chunk(2, dim=0), noise.chunk(2, dim=0)
                
                # DreamBooth Loss
                db_loss = torch.nn.functional.mse_loss(chunk_pred[0].float(), chunk_noise[0].float()) + args.prior_loss_weight * torch.nn.functional.mse_loss(chunk_pred[1].float(), chunk_noise[1].float())
                
                # Reconstruct x0
                alphas = noise_scheduler.alphas_cumprod.to(device)[timesteps]
                pred_x0 = (noisy_latents - (1 - alphas).sqrt().reshape(-1, 1, 1, 1) * noise_pred) / alphas.sqrt().reshape(-1, 1, 1, 1)
                
                # VAE Decode (Gradients Allowed)
                pred_img = vae.decode(pred_x0.chunk(2)[0] / vae.config.scaling_factor, return_dict=False)[0]
                
                # Normalization Fix: [-1, 1] -> [0, 1]
                pred_img = (pred_img / 2 + 0.5).clamp(0, 1)
                
                # Dice Loss
                dice_loss = dice_loss_fn(seg_model(transforms.functional.rgb_to_grayscale(pred_img)), batch["masks"].to(device))
                total_loss = db_loss + args.dice_loss_weight * dice_loss
                
                writer.add_scalar("Loss/total", total_loss.item(), global_step)
                writer.add_scalar("Loss/dice", dice_loss.item(), global_step)

            scaler.scale(total_loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); lr_scheduler.step(); optimizer.zero_grad()

            pbar.set_postfix(loss=total_loss.item()); pbar.update(1); global_step += 1
        
        # Save
        save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
        pipeline.save_pretrained(save_path)
        torch.save({'epoch': epoch+1, 'global_step': global_step, 'unet': unet.state_dict(), 'text_encoder': text_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()}, os.path.join(save_path, "training_state.pth"))

    print("Saving final..."); pipeline.save_pretrained(args.output_dir); writer.close()

if __name__ == "__main__":
    main()
