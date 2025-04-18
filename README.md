# S2S-ST
Implement of "Sparser2Sparse: Single-shot Sparser-to-Sparse Learning for Spatial Transcriptomics Imputation with Natural Image Co-learning"

# Data
All data are downlaoded from HST-1K. Refer [HEST‑1k Dataset on Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest).

# Training
Use `S2S-ST_training.py` to train the model.
Fro example:
```bash
python -u -W ignore S2S-ST_training.py \
--cuda_index 2 --config ./config/config_stride2_64x64_RDN_small.yaml \
--common_markers_path ./Xenium_common_markers.txt \
--batch_size 16 --min_mask_rate 0.5 --train_ratio 1. --gene_scale 10. --gene_loss_rate 10. --gradient_loss_rate 0. --img_grad "" --drop_rate 0. \
--two_step_predict "True" --down_sample_method1 upleft --train_on_origin_size "" \
--img_co_train "True" --split_train "" --train_on_img_only "" --real_LR "" \
--model_note "" --cv upleft \
--marker ERBB2 --model RDN_HABs_M_DCs --Clayers 3 \
--train_dataset TENX94 \
--epochs 3000 --change_lr_to 0.0001 \
--checkpoint_path ""
```