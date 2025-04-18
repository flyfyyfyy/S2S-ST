# S2S-ST

Implementation of **"Sparser2Sparse: Single-shot Sparser-to-Sparse Learning for Spatial Transcriptomics Imputation with Natural Image Co-learning"**

---

## 📊 Data

### 🧬 Spatial Transcriptomics (ST) Data

All ST data are downloaded from the [HEST‑1K Dataset on Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest).

### 🖼️ Natural Image Data

You can download the DIV2K dataset from [this link](https://www.dropbox.com/s/41sn4eie37hp6rh/DIV2K_x2.h5?dl=0).

---

## 🚀 Training

You can train the model using the script `S2S-ST_training.py`.

Example command:
```bash
python -u -W ignore S2S-ST_training.py \
--cuda_index 0 --config ./config/config_stride2_64x64_RDN_small.yaml \
--batch_size 16 --min_mask_rate 0.5 --train_ratio 1. --gene_scale 10 \ 
--gene_loss_rate 10. --gradient_loss_rate 0. --drop_rate 0. \
--two_step_predict "True" --down_sample_method1 upleft \
--img_co_train "True" --split_train "" \
--model_note "" --cv upleft \
--marker ERBB2 --model RDN_HABs_M_DCs --Clayers 3 \
--train_dataset TENX94 \
--epochs 3000 --change_lr_to 0.0001 \
--checkpoint_path ""
```
Make sure to adjust the parameters as needed for your specific dataset and GPU setup.