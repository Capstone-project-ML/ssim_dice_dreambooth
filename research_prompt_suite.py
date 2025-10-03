# research_prompt_suite.py
#
# A comprehensive script for qualitative research experiments and stress-testing
# on fine-tuned medical imaging models (e.g., Stable Diffusion with DreamBooth).
# This script is designed to systematically evaluate model capabilities across
# clinical, compositional, and adversarial prompts.

import argparse
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import re

def parse_args():
    """Parses command-line arguments for the experiment script."""
    parser = argparse.ArgumentParser(description="Run a research-grade prompt suite on a fine-tuned model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (e.g., './results_dreambooth_dice')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./research_suite_output",
        help="Base directory to save the generated images."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['dice', 'ssim', 'combined'],
        help="The type of model being tested, used for organizing output files."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible image generation."
    )
    return parser.parse_args()

def sanitize_filename(text):
    """Creates a safe filename from a prompt string."""
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace spaces with underscores, make lowercase, and shorten
    return text.lower().replace(' ', '_')[:100]

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- RESEARCH PROMPT SUITE ---
    # This list of experiments is designed to be used for all <nih-xray> DreamBooth models.
    research_experiments = [
        # === 1. Baseline Fidelity & Quality ===
        {"prompt": "a photo of <nih-xray>"},
        {"prompt": "a high-quality photo of <nih-xray>"},
        {"prompt": "a high-resolution x-ray of <nih-xray>"},
        {"prompt": "a clear photo of <nih-xray> showing the heart and lungs"},
        {"prompt": "a noisy, low-quality photo of <nih-xray>"},

        # === 2. Clinical & Pathological Prompts (based on NIH ChestX-ray14 dataset pathologies) ===
        # --- Absence of disease ---
        {"prompt": "a photo of a normal <nih-xray>"},
        {"prompt": "a photo of an unremarkable <nih-xray>"},
        # --- Presence of specific diseases ---
        {"prompt": "a photo of <nih-xray> with signs of Atelectasis"},
        {"prompt": "a photo of <nih-xray> showing Cardiomegaly"},
        {"prompt": "a photo of <nih-xray> with Consolidation"},
        {"prompt": "a photo of <nih-xray> with Edema"},
        {"prompt": "a photo of <nih-xray> showing Effusion"},
        {"prompt": "a photo of <nih-xray> with Emphysema"},
        {"prompt": "a photo of <nih-xray> with Fibrosis"},
        {"prompt": "a photo of <nih-xray> with a Hernia"},
        {"prompt": "a photo of <nih-xray> showing Infiltration"},
        {"prompt": "a photo of <nih-xray> with a Mass"},
        {"prompt": "a photo of <nih-xray> with a Nodule"},
        {"prompt": "a photo of <nih-xray> with Pleural Thickening"},
        {"prompt": "a photo of <nih-xray> with signs of Pneumonia"},
        {"prompt": "a photo of <nih-xray> showing Pneumothorax"},
        # --- Combinations ---
        {"prompt": "a photo of <nih-xray> showing both Atelectasis and Effusion"},
        {"prompt": "a photo of <nih-xray> with Cardiomegaly and signs of Pneumonia"},

        # === 3. Viewpoint & Compositionality ===
        # --- Viewpoints ---
        {"prompt": "a photo of <nih-xray>, AP view"},
        {"prompt": "a photo of <nih-xray>, PA view"},
        {"prompt": "a photo of <nih-xray>, lateral view"},
        # --- Demographics ---
        {"prompt": "a pediatric photo of <nih-xray>"},
        {"prompt": "a photo of <nih-xray> from an elderly patient"},
        # --- Added Objects ---
        {"prompt": "a photo of <nih-xray> with a pacemaker"},
        {"prompt": "a photo of <nih-xray> with a chest tube"},
        {"prompt": "a photo of <nih-xray> with EKG leads visible"},

        # === 4. Style & Abstraction (Testing Robustness) ===
        {"prompt": "a watercolor painting of <nih-xray>"},
        {"prompt": "a charcoal sketch of <nih-xray>"},
        {"prompt": "a 3D rendering of <nih-xray>"},
        {"prompt": "an anatomical diagram of <nih-xray>"},
        {"prompt": "a blueprint of <nih-xray>"},

        # === 5. Stress & Adversarial Tests ===
        # --- Unrelated Concepts (should ignore <nih-xray>) ---
        {"prompt": "a photo of a cat"},
        {"prompt": "a photo of a dog, <nih-xray>"},
        {"prompt": "a beautiful landscape painting"},
        # --- Contradictory Concepts ---
        {"prompt": "a photo of <nih-xray> made of wood"},
        {"prompt": "a colorful photo of <nih-xray>"},
        {"prompt": "a photo of <nih-xray> with a smiling face"},
        # --- Negative Prompts ---
        {"prompt": "a photo of <nih-xray>", "negative_prompt": "bones, spine, ribs"},
        {"prompt": "a photo of <nih-xray>", "negative_prompt": "heart, lungs"},
        {"prompt": "a photo of <nih-xray> with a pacemaker", "negative_prompt": "wires, leads"},
        {"prompt": "a photo of <nih-xray>", "negative_prompt": "blur, artifacts, noise, low quality"},
    ]
    
    # --- SCRIPT EXECUTION ---
    
    # Create the output directory based on the model type
    output_path = Path(args.output_dir) / args.model_type
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Selected model type: {args.model_type}")
    print(f"Saving images to: {output_path}")

    # Load the pipeline
    print(f"Loading model from {args.model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Run all defined experiments
    for i, exp in enumerate(research_experiments):
        prompt = exp.get("prompt")
        neg_prompt = exp.get("negative_prompt")
        
        print(f"\n({i+1}/{len(research_experiments)}) Generating for prompt: '{prompt}'")
        if neg_prompt:
            print(f"  -> Negative prompt: '{neg_prompt}'")
        
        pipeline_args = {"prompt": prompt, "generator": generator}
        if neg_prompt:
            pipeline_args["negative_prompt"] = neg_prompt
            
        with torch.no_grad():
            image = pipeline(**pipeline_args).images[0]
            
        # Save the image
        filename_base = f"{i:03d}_{sanitize_filename(prompt)}"
        if neg_prompt:
            filename_base += "_neg_" + sanitize_filename(neg_prompt)
        
        image_save_path = output_path / f"{filename_base}.png"
        image.save(image_save_path)
        print(f"  -> Image saved to {image_save_path}")

    print("\nAll research experiments complete!")

if __name__ == "__main__":
    main()