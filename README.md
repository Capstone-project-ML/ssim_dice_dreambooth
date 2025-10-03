CheXmask: https://github.com/ngaggion/CheXmask-Database?tab=readme-ov-file
(Download the segmentation model weights from here)

NIH Binary Segmentation Masks: https://www.kaggle.com/datasets/smeraa/nih-sample-segmentation-masks

Info about the 3 finetuning codes: https://docs.google.com/spreadsheets/d/1I_K_XJViRuMjK47kHTt1ADlrx5gD3Z_vMITQRlgAqWM/edit?usp=sharing

Link to segmentarion model: https://huggingface.co/vanillacoke/segmentation/blob/main/best_segmentation_model_50_epochs.pth

command to run evaluate.py:
python evaluate.py \
  --model_path "./results_dreambooth_dice/final_model" \
  --real_images_path "./sample/images" \
  --prompt "a photo of <nih-xray>" \
  --num_samples 10 \
  --calculate_extended_metrics \
  --seg_model_weights "./path/to/your/seg_model.pth" \
  --real_masks_path "./path/to/your/xray_masks"
