import os
import sys
import argparse
import shutil
import copy

import random
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torchvision import transforms

import models
import utils
import unet_utils

import importlib
importlib.reload(models)
importlib.reload(utils)
importlib.reload(unet_utils)

from models import *
from utils import *
from unet_utils import *

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        return False  

parser = argparse.ArgumentParser()

parser.add_argument('--cuda_index', type=int, default=0, help='cuda index to use')
parser.add_argument('--config', type=str, default = './config/config_example.yaml', help='path to the config file')
parser.add_argument('--marker', type=str, default = None, help='marker to use, "," separated')
parser.add_argument("--remove_batch_effect", type=str_to_bool, default = False, help="remove batch effect")
parser.add_argument("--train_dataset", type=str,default=None ,help="dataset to use")
parser.add_argument("--train_ratio", type=float, default = 0.5, help="train ratio")
parser.add_argument("--target_level", type=int, default = None, help="target level")
parser.add_argument("--gradient_loss_rate", type=float, default = 0, help="gradient loss rate")
parser.add_argument("--img_grad", type=str_to_bool, default = False, help="img grad")
parser.add_argument("--gene_scale", type=float, default = 10., help="gene scale")
parser.add_argument("--gene_loss_rate", type=float, default = 10., help="gene scale rate")
parser.add_argument("--cv", type=str, default = "upleft", help="cross validation, remained points")
parser.add_argument("--train_on_origin_size", type=str_to_bool, default = False, help="train on origin size")
parser.add_argument("--img_co_train", type=str_to_bool, default = False, help="img_co_train")
parser.add_argument("--split_train", type=str_to_bool, default = False, help="split train for img_co_train or not")
parser.add_argument("--train_on_img_only", type=str_to_bool, default = False, help="train on img only")
parser.add_argument("--real_LR", type=str_to_bool, default = False, help="real LR")
parser.add_argument("--two_step_predict", type=str_to_bool, default = False, help="two step predict")
parser.add_argument("--down_sample_method1", type=str, default = "upleft", help="down sample method1")
parser.add_argument("--test_dataset", type=str,default=None ,help="dataset to use")
parser.add_argument("--metric_dataset", type=str,default=None ,help="dataset to use")
parser.add_argument("--min_mask_rate", type=float, default = None, help="min mask rate")
parser.add_argument("--drop_rate", type=float, default = 0., help="dropout rate")
parser.add_argument("--batch_size", type=int, default = None, help="batch size")
parser.add_argument('--model', type=str, default="Diffusion", help='model to use')
parser.add_argument('--Clayers', type=int, default=None, help='number of Clayers')
parser.add_argument('--model_note', type=str, default=None, help='model note')
parser.add_argument("--epochs", type=int, default = 0, help="number of epochs")
parser.add_argument("--change_lr_to", type=float, default = 0., help="change learning rate to")
parser.add_argument("--checkpoint_path", type=str, default = None, help="path to the checkpoint file")
parser.add_argument('--scheduler', choices=['cosine', 'step', 'plateau', 'multi_step',"None"],default='cosine', help='scheduler to use')
parser.add_argument('--scheduler_param', type=str, default=None, help='{cosine: eta_min or like "d10" mean lr/10), step: step_size, plateau: patience,multi_step: milestones, like 10,20,30}')

args = parser.parse_args()
device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")

# %%
config = load_yaml(args.config)
# convert config to global variables
def dict2objedct(config):
    for k,v in config.items():
        if isinstance(v, dict):
            dict2objedct(v)
        else:
            globals()[k] = v
dict2objedct(config)

torch.manual_seed(seed+1)
np.random.seed(seed)

# %%
meta_df = pd.read_csv(f"{data_dir}/HEST_v1_1_0.csv")

ST_ids = meta_df[meta_df['st_technology'] == "Spatial Transcriptomics"]['id'].to_list()     # sparse squre matrix
Xenium_ids = meta_df[meta_df['st_technology'] == "Xenium"]['id'].to_list()                  # dense squre matrix
Visium_ids = meta_df[meta_df['st_technology'] == "Visium"]['id'].to_list()                  # dense hexagons matrix
Visium_HD_ids = meta_df[meta_df['st_technology'] == "Visium HD"]['id'].to_list()            # denser squre matrix    # ['TENX156',\n 'TENX155',\n 'TENX154',\n 'TENX153',\n 'TENX146',\n 'TENX145',\n 'TENX144',\n 'TENX131',\n 'TENX129',\n 'TENX128']

# %%
batch_size = batch_size if args.batch_size is None else args.batch_size
min_mask_rate = args.min_mask_rate if args.min_mask_rate is not None else min_mask_rate
drop_rate = args.drop_rate if args.drop_rate is not None else drop_rate
patch_norm_temp = "patch_emb_normal" if patch_emb_normal else "patch_emb_unnormal"

marker = args.marker.split(",") if args.marker else common_genes
args.marker = "all" if args.marker is None else args.marker
marker = common_genes if args.marker in ["all","All","ALL","each","Each","EACH",""] else marker
train_on_each_gene = True if args.marker in ["each","Each","EACH"] else False
    
train_dataset_names = args.train_dataset.split(",") if args.train_dataset else None
print(f"Train Dataset: {train_dataset_names}")
id_type = "ST" if train_dataset_names[0] in ST_ids else "Xenium" if train_dataset_names[0] in Xenium_ids else "Visium" if train_dataset_names[0] in Visium_ids else "Visium_HD" if train_dataset_names[0] in Visium_HD_ids else None

min_mask_rate = min_mask_rate * 0.8 if train_dataset_names[0] in Visium_ids else min_mask_rate
gene_augment = False if (train_dataset_names[0] in Visium_ids)  or ("wo_aug" in args.model_note) else True

adata_path = f"{data_dir}/adata/{train_dataset_names[0]}.h5ad"
print(f"Lodaing adata from {adata_path}")

# %%
if os.path.exists(adata_path):
    adata = sc.read_h5ad(adata_path)
else:
    id = train_dataset_names[0]
    st_path = f'{data_dir}/st/{id}.h5ad'
    patches_path = f'{data_dir}/patches/{id}.h5'
    wsi_path = f'{data_dir}/wsis/{id}.tif'
    adata_path = f"{data_dir}/adata/{id}.h5ad"
    adata__path = f"{data_dir}/adata_/{id}.h5ad"

    from conch.open_clip_custom import create_model_from_pretrained
    if "model" not in globals() and "preprocess" not in globals():
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="your_token")
        model.to(device)

    adata = read_spatial_data(st_path)
    # load patches into adata
    adata = load_patches(adata,patches_path)
    # filter cells
    adata = filter_cells(adata,min_counts=1)
    # log-normalize data
    adata = log_normalize(adata)

    # reset adata index
    if id_type == "ST":
        adata = reset_ST_index(adata)
    elif id_type == "Xenium":
        if "array_row" not in adata.obs.columns or "array_col" not in adata.obs.columns:
            adata = reset_Xenium_columns(adata)
            
    # patch image embedding
    adata.obsm["patch_emb"] = np.zeros((adata.shape[0], 512))
    # image embedding
    for i in range(adata.shape[0]):
        print(f"Processing {id} - {i}/{adata.shape[0]}",end="\r")
        image = Image.fromarray(adata.obsm["patch_image"][i])
        image = preprocess(image).unsqueeze(0)
        image = image.to(device)
        with torch.inference_mode():
            ###########################################################################################################################
            image_embs = model.encode_image(image, proj_contrast=False, normalize=patch_emb_normal)# scale to match gene expression range
            ###########################################################################################################################
        adata.obsm["patch_emb"][i] = image_embs.squeeze().detach().cpu().numpy()

    # add a empty patch to avoid error, used for -1 index    
    import anndata
    new_X = np.concatenate([adata.X,np.zeros((1, adata.shape[1])).astype(np.float32)],axis=0)
    adata_ = anndata.AnnData(X=new_X, var=adata.var.copy())
    adata_.obsm["patch_emb"] = np.concatenate([adata.obsm["patch_emb"],np.zeros((1, 512)).astype(np.float32)],axis=0)
    adata_.obsm["patch_image"] = np.concatenate([adata.obsm["patch_image"],np.zeros((1, 224, 224, 3)).astype(np.uint8)],axis=0)
    adata_.obsm["spatial"] = np.concatenate([adata.obsm["spatial"],np.array([[np.nan, np.nan]]).astype(np.float32)],axis=0)
    try:
        adata.write(adata_path)
        adata_.write(adata__path)
    except:
        print(f"Failed to save {id} adata")
    
if args.model == "ViT":
    adata = adata[:,adata.var["n_cells_by_counts"]>len(adata.obs.index)*0.1]


# %%
marker_index = [adata.var_names.tolist().index(m) for m in marker]
print(f"Using marker: {marker} Marker index: {marker_index}")

row_max,col_max = (adata.obs["array_row"].max()+1)+(adata.obs["array_row"].max()+1)%2 , (adata.obs["array_col"].max()+1)+(adata.obs["array_col"].max()+1)%2

gene_exp_matrix = torch.zeros(adata.shape[1],row_max,col_max)
patch_img_m = torch.zeros(row_max,col_max,224,224,3)
mask_whole = torch.zeros(row_max,col_max)
for obs in range(adata.shape[0]):
    gene_exp_matrix[:,adata.obs["array_row"].iloc[obs],adata.obs["array_col"].iloc[obs]] = torch.from_numpy(adata.X[obs])
    patch_img_m[adata.obs["array_row"].iloc[obs],adata.obs["array_col"].iloc[obs]] = torch.from_numpy(adata.obsm["patch_image"][obs])
    mask_whole[adata.obs["array_row"].iloc[obs],adata.obs["array_col"].iloc[obs]] = 1

gs = []
for i,j in np.ndindex(row_max-patch_size,col_max-patch_size):
    if mask_whole[i:i+64,j:j+64].sum() > min_mask_rate*patch_size**2:
        gs.append([i,j])

# %%
if args.cv == "upleft":
    x_,y_ = 0,0
elif args.cv == "upright":
    x_,y_ = 0,1
elif args.cv == "downleft":
    x_,y_ = 1,0
elif args.cv == "downright":
    x_,y_ = 1,1

train_gs, test_gs = [], []
gap = int(args.train_ratio) if int(args.train_ratio) > stride else stride
for g in gs:    
    x,y = g
    if int(x) % gap == x_ and int(y) % gap == y_:
        mask = mask_whole[x:x+patch_size,y:y+patch_size]
        if mask.sum() >= min_mask_rate * mask.numel():
            train_gs.append(g)
print(f"Totel: {len(gs)}, Train: {len(train_gs)}")

# %%
img_h5_file = f"./data/DIV2K_HR.h5" if not args.real_LR else './data/DIV2K_x2.h5'

# creat new real img dataset
'''
from torchvision.transforms import Grayscale
transform = transforms.Compose([
    transforms.ToTensor(),
    Grayscale(num_output_channels=1)
    ]
)
with h5py.File('./data/DIV2K_x2.h5', 'r') as f:
    with h5py.File("DIV2K_HR.h5", 'w') as f2:
        f2.create_group("hr")
        f2.create_group("lr")
        for i,k in enumerate(f["lr"]):
            hr = transform(f["hr"][k][:])
            f2["hr"].create_dataset(k,data=hr.detach().cpu().numpy())
            lr = transform(f["lr"][k][:])
            f2["lr"].create_dataset(k,data=lr.detach().cpu().numpy())
'''

class Img_Dataset(Dataset):
    def __init__(self, h5_file, patch_size, stride,down_sample_method="upleft",drop_rate=0.):
        super(Img_Dataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.stride = stride
        self.down_sample_method = down_sample_method
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.patch_size * 2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.CenterCrop(self.patch_size),
            # transforms.Lambda(lambda img: img + 0.1 * torch.randn_like(img))
        ])
        self.device = device
        self.drop_rate = drop_rate
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            img = torch.from_numpy(f['hr'][str(idx)][:])
            hr = self.transform(img).unsqueeze(0)
            hr = torch.clamp(hr, 0, 1)
            lr = down_sample(hr, self.stride, method=self.down_sample_method)
            if self.drop_rate > 0:
                mask = torch.bernoulli(torch.ones_like(lr) * (1 - self.drop_rate))
                lr = lr * mask
            else:
                mask = torch.ones_like(lr)
            data = Data(target=hr, LR=lr,idx=torch.tensor(idx),mask_DC = mask)
            return data

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr'])

# %%
class Img_Dataset2(Dataset):
    def __init__(self, h5_file, patch_size, scale,down_sample_method = "upleft",drop_rate=0.):
        super(Img_Dataset2, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale
        self.down_sample_method = down_sample_method
        self.drop_rate = drop_rate
        from torchvision.transforms import Grayscale
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            Grayscale(num_output_channels=1)
            ]
        )
            
    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_rotate_90(lr, hr)
            lr = self.transform(lr).unsqueeze(0)
            hr = self.transform(hr).unsqueeze(0)
            
            if self.down_sample_method == "upleft":
                lr = up_sample(lr, self.scale)
            if self.drop_rate > 0:
                mask = torch.bernoulli(torch.ones_like(lr) * (1 - self.drop_rate))
                lr = lr * mask
            else:
                mask = torch.ones_like(lr)
            
            data = Data(target=hr, LR=lr,idx=torch.tensor(idx),mask_DC = mask)
            return data

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# %%
def model_cls(model_name):
    if model_name in ["RDN","RDN_HAB","RDN_HABs","RDN_Multi_Scale","RDN_SPANs","RDN_SPANs2","ViT"]:
        model_cls = True
    elif model_name in ["UNet","Unet","Unet_DCs","UNet_DCs","RDN_M","RDN_M_DCs","RDNet","RDN_HAB_M_DCs","RDN_HABs_M_DCs","RDN_SPANs_M","RDN_SPANs_M_DCs","RDN_HABs_M2","RDN_HABs_M2_DCs","RDN_HABs_M3","RDN_HABs_M3_DCs"]:
        model_cls = False
    return model_cls


class Gene_Dataset(Dataset):
    def __init__(self, gene_exp_matrix, mask_whole, upleft_idx,down_sample_method="upleft",gene_augment=True,
                 model="RDN",marker_idx = [7], stride=1, gene_scale = 10,drop_rate=0.,patch_size = 64,
                 train_on_origin_size = False,training = False,
                 img_dataset = None):
        super(Gene_Dataset, self).__init__()
        self.gene_exp_matrix = gene_exp_matrix
        self.mask_whole = mask_whole
        if model == "ViT":
            pad_w = (64 - (gene_exp_matrix.shape[1] % 64)) % 64  
            pad_h = (64 - (gene_exp_matrix.shape[2] % 64)) % 64  
            self.gene_exp_matrix = F.pad(self.gene_exp_matrix, (0, pad_h, 0, pad_w), mode='constant', value=0)
            self.mask_whole = F.pad(self.mask_whole,(0, pad_h, 0, pad_w), mode='constant', value=0)
        self.upleft_idx = upleft_idx
        self.down_sample_method = down_sample_method
        self.gene_augment = gene_augment
        self.model = model
        
        self.marker_idx = marker_idx if isinstance(marker_idx, list) else [marker_idx]
        self.stride = stride
        self.train_on_origin_size = train_on_origin_size
        self.training = training
        self.img_dataset = img_dataset
        self.gene_scale = gene_scale
        self.drop_rate = drop_rate
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.upleft_idx)
    
    def __getitem__(self, idx):
        x,y = self.upleft_idx[idx]
        data = Data()

        if self.model != "ViT":
            HR = self.gene_exp_matrix[self.marker_idx,x:x+self.patch_size,y:y+self.patch_size].unsqueeze(0)
            mask = self.mask_whole[x:x+self.patch_size,y:y+self.patch_size].unsqueeze(0)
        elif self.model == "ViT":
            if self.training:
                # HR = self.gene_exp_matrix[torch.randint(self.gene_exp_matrix.shape[0],(1,)),:,:].unsqueeze(0)
                HR = self.gene_exp_matrix[self.marker_idx,:,:].unsqueeze(0)
            else:
                HR = self.gene_exp_matrix[self.marker_idx,:,:].unsqueeze(0)
            mask = self.mask_whole.unsqueeze(0)

        if self.model != "ViT":
            if self.training and self.gene_augment:
                r_ = torch.rand(3)
                HR = torch.flip(HR,[2]) if r_[0] > 0.5 else HR
                mask = torch.flip(mask,[1]) if r_[0] > 0.5 else mask
                
                HR = torch.flip(HR,[3]) if r_[1] > 0.5 else HR
                mask = torch.flip(mask,[2]) if r_[1] > 0.5 else mask
                
                HR = torch.rot90(HR,1,[2,3]) if r_[2] > 0.5 else HR
                mask = torch.rot90(mask,1,[1,2]) if r_[2] > 0.5 else mask
                
                r__ = random.choice([0, 1, 2, 3])
                HR = torch.rot90(HR, r__, [2, 3])
                mask = torch.rot90(mask, r__, [1, 2])
                
                # HR = HR + torch.randn_like(HR) * 0.1
                # HR = torch.clamp(HR, 0, 1)

        data.marker_idx = torch.tensor(self.marker_idx).unsqueeze(0).long()
        data.mask = mask
        data.upleft = torch.tensor([x,y]).unsqueeze(0).long()
        
        if self.training:
            if self.train_on_origin_size:
                HR1 = down_sample(HR, self.stride, method="topleft")
                data.target = up_sample(HR1, self.stride, method="zeros")
                data.mask = down_sample(data.mask, self.stride, method="zeros")
                if model_cls(self.model):
                    data.LR = down_sample(HR1, self.stride, method=self.down_sample_method)
                else:
                    data.LR = up_sample(down_sample(HR1, self.stride, method=self.down_sample_method), self.stride, method=self.down_sample_method)
                    
            else:
                data.target = down_sample(HR, self.stride, method="topleft")
                data.mask = down_sample(data.mask, self.stride, method="topleft")
                if model_cls(self.model):
                    data.LR = down_sample(data.target, self.stride, method="topleft")
                else:
                    data.LR = down_sample(data.target, self.stride, method=self.down_sample_method)
        else:
            data.target = HR
            if model_cls(self.model):
                data.LR = down_sample(data.target, self.stride, method="topleft")
            else:
                data.LR = down_sample(data.target, self.stride, method=self.down_sample_method)
        
        data.LR = data.LR / self.gene_scale
        if self.drop_rate > 0:
            mask = torch.bernoulli(torch.ones_like(data.LR) * (1 - self.drop_rate))
            data.LR = data.LR * mask
        else:
            mask = torch.ones_like(data.LR)
        data.mask_DC = mask
        
        data_img = Data()
        if self.img_dataset is not None:
            data_img = self.img_dataset[random.randint(0,len(self.img_dataset)-1)]
        return data, data_img

# %%
img_dataset = Img_Dataset(img_h5_file, patch_size, stride,down_sample_method="topleft" if model_cls(args.model) else args.down_sample_method1,drop_rate=args.drop_rate) if (args.img_co_train or args.train_on_img_only) else None
if args.real_LR:
    img_dataset = Img_Dataset2(img_h5_file, patch_size//stride, stride,down_sample_method=False if model_cls(args.model) else  args.down_sample_method1,drop_rate=args.drop_rate) if (args.img_co_train or args.train_on_img_only) else None

train_loader = DataLoader(Gene_Dataset(gene_exp_matrix,mask_whole,train_gs,down_sample_method=args.down_sample_method1,gene_augment = gene_augment,
                                            model=args.model, marker_idx=marker_index, stride=stride, gene_scale=args.gene_scale,drop_rate=args.drop_rate,patch_size=patch_size,
                                            train_on_origin_size=args.train_on_origin_size,training=True,
                                            img_dataset=img_dataset), 
                               batch_size=16, shuffle=True, num_workers=4)

test_loader = DataLoader(Gene_Dataset(gene_exp_matrix,mask_whole,train_gs,down_sample_method=args.down_sample_method1,gene_augment = False,
                                           model=args.model, marker_idx=marker_index,stride=stride,gene_scale=args.gene_scale,patch_size=patch_size,
                                           training=False), 
                              batch_size=16, shuffle=False, num_workers=4)

print(f"Image H5 file: {img_h5_file}")
print("Batch size: ",batch_size)

# %% [markdown]
# # Training

# %%
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

class DataConsistencyLayer(nn.Module):
    def __init__(self, noise_lvl=None):
        super(DataConsistencyLayer, self).__init__()
        self.noise_lvl = noise_lvl# noise level is lambda in the paper

    def forward(self, k, k0, mask):
        return data_consistency(k, k0, mask, self.noise_lvl)


class Unet_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,**kwargs):
        super(Unet_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        
        for i in range(Clayers):
            self.nets.append(UNet(
                # UNet parameters
                in_channel=1,
                out_channel=1,
                inner_channel=64,
                norm_groups=32,
                channel_mults=[1, 2, 4, 8, 8],
                with_noise_level_emb=True,
                use_affine_level=False,
                input_FWA = False,
                # Resnet parameters
                res_blocks=3,
                dropout=0,
                # Conditions
                conditions = ["LR"],
                concat_ori_LR=True,
                concat_ori_patch_emb=False,
                # x embedding parameters
                x_emb_out_channel=None,
                x_emb_layers=0,
                # Patch Embedding parameters
                patch_emb_in_channel=512,
                patch_emb_out_channel=None,
                patch_emb_layers=0,
                # Attention parameters
                attn_res=[1,2,4,8], # layer to use attention, the numbers in channel_mults
                patch_emb_cross_attn=False,
                patch_emb_concat=False,
                gene_emb_cross_attn=False,
                gene_emb_concat=True,
                n_head=4,
                head_dim = None,
                mlp_ratio=None,
                # marker_idx embedding parameters
                num_classes=8,
                # LR embedding parameters
                LR_emb_out_channel=None,
                LR_emb_layers=0,
                # LR and patch cross
                LR_patch_cross_attn=False
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))

    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, UNet):
                data.LR = layer(None,data) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs


class RDN_M_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,**kwargs):
        super(RDN_M_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        
        for i in range(Clayers):
            self.nets.append(RDN_M(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_M):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs

class RDN_HAB_M_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,window_size = 8,**kwargs):
        super(RDN_HAB_M_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        self.window_size = window_size
            
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_HAB_M(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
                window_size = window_size
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_HAB_M):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs
    
class RDN_HABs_M_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,window_size = 8,**kwargs):
        super(RDN_HABs_M_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        self.window_size = window_size
        
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_HABs_M(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
                window_size = window_size
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_HABs_M):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs

class RDN_HABs_M2_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,window_size = 8,**kwargs):
        super(RDN_HABs_M2_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        self.window_size = window_size
        
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_HABs_M2(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
                window_size = window_size
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_HABs_M2):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs
    
class RDN_HABs_M3_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,window_size = 8,**kwargs):
        super(RDN_HABs_M3_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        self.window_size = window_size
        
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_HABs_M3(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
                window_size = window_size
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_HABs_M3):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs

class RDN_SPANs_M_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,**kwargs):    
        super(RDN_SPANs_M_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_SPANs_M(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_SPANs_M):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs
    
class RDN_SPANs2_M_DCs(nn.Module):
    def __init__(self,Clayers=5,DC = True,noise_lvl=None,num_features=64,num_blocks=10,num_layers=6,growth_rate=64,scale_factor=2,num_channels=1,**kwargs):
        super(RDN_SPANs2_M_DCs, self).__init__()
        self.noise_lvl = noise_lvl
        self.nets = nn.ModuleList()
        self.Clayers = Clayers
        self.scale_factor = scale_factor
        self.num_channels = num_channels        
        G0, D, C, G = num_features, num_blocks, num_layers, growth_rate
        self.G0, self.D, self.C, self.G = G0, D, C, G
        
        self.nets.append(DataConsistencyLayer(noise_lvl))
        
        for i in range(Clayers):
            self.nets.append(RDN_SPANs2_M(
                scale_factor=scale_factor, 
                num_channels=num_channels, 
                num_features=G0,
                num_blocks=D,
                num_layers=C,
                growth_rate=G,
            ))
            if DC:
                if i != Clayers-1:
                    self.nets.append(DataConsistencyLayer(noise_lvl))
    
    def forward(self, data,mask):
        data.LR_raw = data.LR.clone()
        rs = []
        for layer in self.nets:
            if isinstance(layer, RDN_SPANs2_M):
                data.LR = layer(data.LR) + data.LR
                rs.append(data.LR)
            elif isinstance(layer, DataConsistencyLayer):
                data.LR = layer(data.LR, data.LR_raw, mask)
        return rs

# %%
import torch
import math

class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=16, in_channel=1, embed_dim=1024):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # conv: (b, 1, h, w) -> (b, 1024, h/16, w/16)
        # flatten: (b, 256, h/16, w/16) -> (b, 256, hw/256)
        # reshape:(b, 256, hw/256) -> (b, hw/256, 256) (batch, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(torch.nn.Module):
    #dim = patch_size * patch_size * 4, which is quadrupled in PatchEmbed.
    def __init__(self, dim=16, num_heads=8, drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim*3, bias=True)
        self.drop1 = torch.nn.Dropout(drop_ratio)
        self.proj = torch.nn.Linear(dim, dim)
        self.drop2 = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(D / self.num_heads)).permute(2, 0, 3, 1, 4)
        # (batch, num_heads, num_patches, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, num_patches, num_patches)
        att = att.softmax(dim=-1)
        att = self.drop1(att)

        x = (att @ v).transpose(1, 2).flatten(2)  # B,N,dim
        x = self.drop2(self.proj(x))
        return x

class Mlp(torch.nn.Module):
    def __init__(self, in_dim=1024, drop_ratio=0.):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, in_dim*2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(in_dim*2, in_dim)
        self.drop = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, in_dim=1024, num_heads=8, drop_ratio=0.):
        super(Block, self).__init__()
        # This step is very important, otherwise it will be difficult to converge.
        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.attn = Attention(dim=in_dim, num_heads=num_heads, drop_ratio=drop_ratio)
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = Mlp(in_dim=in_dim,drop_ratio=drop_ratio)
        self.drop = torch.nn.Dropout(0.)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

def absolute_position_encoding(seq_len, embed_dim):
    """
    Generate absolute position coding
    :param seq_len: Sequence length
    :param embed_dim: PatchEmbed length
    :return: absolute position coding
    """
    # (10000 ** ((2 * i) / embed_dim))
    seq_len = int(seq_len)
    pos_enc = torch.zeros((seq_len, embed_dim))
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (2*i / embed_dim)))
            if i + 1 < embed_dim:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (2*i / embed_dim)))
    return pos_enc

class VisionTransformer(torch.nn.Module):
    def __init__(self, patch_size=16, in_c=1, embed_dim=1024, depth=12, num_heads=8, drop_ratio=0.):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channel=in_c, embed_dim=embed_dim)
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # depth transformer code blocks.
        self.blocks = torch.nn.Sequential(*[
            Block(in_dim=embed_dim, num_heads=num_heads, drop_ratio=drop_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        b, _, h, w = x.shape
        num_patches = (h / self.patch_size) * (w / self.patch_size)
        pri_x = x
        pos = absolute_position_encoding(num_patches, self.embed_dim).to(device)
        x = self.patch_embed(x)
        # Add position encoding.
        x = self.pos_drop(x + pos)
        # （batch, num_patches, embed_dim）
        x = self.blocks(x)
        # (b, 4, dim, ph, pw)
        x = x.reshape(b, -1, int(self.embed_dim//4), 4).transpose(1, 3).reshape(b, 4, int(self.embed_dim//4), int(h / self.patch_size), int(w / self.patch_size))
        fina_x = torch.zeros((b, 4, h, w)).to(device)
        k = 0
        # Reverse PatchEmbedding.
        for i, j in np.ndindex(self.patch_size, self.patch_size):
            fina_x[:, :, i::self.patch_size, j::self.patch_size] = x[:, :, k, :, :]
            k += 1
        x = pri_x + fina_x
        x = fina_x
        
        x = rearrange(x, 'b (h w) p1 p2 -> b 1 (p1 h) (p2 w)', h=2, w=2)
        return x

# %%
ps_ = {"scale_factor": stride,
        "num_channels": 1,
        "num_features": G0,
        "num_blocks": D,
        "num_layers": C,
        "growth_rate": G,
        "window_size": window_size,
        }

if "RDN_" in args.model:
    if "DCs" in args.model:
        assert args.Clayers is not None and args.Clayers>0, "args.Clayers is None or <=0"
        model = eval(args.model)(Clayers=args.Clayers,DC = True,noise_lvl=None,**ps_).to(device)
    else:
        model = eval(args.model)(**ps_).to(device)
elif args.model == "RDN":
    model = RDN(**ps_).to(device)


if args.model == "Unet":
    model = UNet(
        # UNet parameters
        in_channel=1,
        out_channel=1,
        inner_channel=64,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        with_noise_level_emb=True,
        use_affine_level=False,
        input_FWA = False,
        # Resnet parameters
        res_blocks=3,
        dropout=0,
        # Conditions
        conditions = ["LR"],
        concat_ori_LR=True,
        concat_ori_patch_emb=False,
        # x embedding parameters
        x_emb_out_channel=None,
        x_emb_layers=0,
        # Patch Embedding parameters
        patch_emb_in_channel=512,
        patch_emb_out_channel=None,
        patch_emb_layers=0,
        # Attention parameters
        attn_res=[1,2,4,8], # layer to use attention, the numbers in channel_mults
        patch_emb_cross_attn=False,
        patch_emb_concat=False,
        gene_emb_cross_attn=False,
        gene_emb_concat=True,
        n_head=4,
        head_dim = None,
        mlp_ratio=None,
        # marker_idx embedding parameters
        num_classes=8,
        # LR embedding parameters
        LR_emb_out_channel=None,
        LR_emb_layers=0,
        # LR and patch cross
        LR_patch_cross_attn=False
    ).to(device)

elif args.model == "RDNet":
    model = RDNet(
        num_init_features=32,# 64
        patch_size = 1,
        growth_rates=(32,52,64,64,64,64,112),# (64, 104, 128, 128, 128, 128, 224),
        num_blocks_list=(3, 3, 3, 3, 3, 3, 3),
        bottleneck_width_ratio=4,
        zero_head=False,
        in_chans=1,  # timm option [--in-chans]
        num_classes=0,  # timm option [--num-classes]
        drop_rate=0.,  # timm option [--drop: dropout ratio]
        drop_path_rate=0.,  # timm option [--drop-path: drop-path ratio]
        checkpoint_path=None,  # timm option [--initial-checkpoint]
        transition_compression_ratio=0.5,
        ls_init_value=1e-6,
        is_downsample_block=(None, False, False, False, False, False, False),
        block_type="Block",
        head_init_scale=1.,
    ).to(device)

elif args.model == "ViT":
    # patch_size = 8,embed_dim = 4*8*8,depth = 8,num_heads = 8,drop_ratio = 0.,
    # patch_size = 16,embed_dim = 4*16*16,depth = 12,num_heads = 8,drop_ratio = 0.,
    model = VisionTransformer(patch_size=8, in_c=1, embed_dim=4*8*8, depth=8, num_heads=8, drop_ratio=0.).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=args.change_lr_to if args.change_lr_to is not None else 0.0001, betas=(0.5, 0.999))

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss2D(nn.Module):
    def __init__(self):
        super(GradientLoss2D, self).__init__()
        # Define Sobel kernels for 2D gradient computation
        self.sobel_kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0) / 8.0  # Normalize kernel
        self.sobel_kernel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0) / 8.0  # Normalize kernel

        # Register kernels as buffers so they are not updated during training
        self.register_buffer('sobel_x', self.sobel_kernel_x)
        self.register_buffer('sobel_y', self.sobel_kernel_y)

    def forward(self, pred, target,mask = None):
        # Ensure inputs are 4D tensors (batch_size, channels, height, width)
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError("Input tensors must be 4D (batch_size, channels, height, width)")

        # Compute gradients using Sobel filters
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)

        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

        # Compute gradient magnitude difference
        grad_diff = (
            torch.abs(grad_pred_x - grad_target_x) +
            torch.abs(grad_pred_y - grad_target_y)
        )

        if mask is not None:
            grad_diff = grad_diff * mask
            return grad_diff.sum() / mask.sum()
            
        return grad_diff.mean()

gradient_loss = GradientLoss2D().to(device)

# %%
def get_pred(model, data,stride,down_sample_method,training = False):
    if type(model).__name__ == "UNet":
        pred = model(None,data)
    elif "RDN_" in type(model).__name__:
        if "DCs" in args.model:
            mask = torch.zeros_like(data.LR).to(device)
            if down_sample_method == "upleft":
                mask[:,:,::stride,::stride] = 1.
            elif down_sample_method == "upright":
                mask[:,:,::stride,1::stride] = 1.
            elif down_sample_method == "downleft":
                mask[:,:,1::stride,0::stride] = 1.
            elif down_sample_method == "downright":
                mask[:,:,1::stride,1::stride] = 1.
            if data.mask_DC.shape[-1] == mask.shape[-1]:
                mask = data.mask_DC * mask
            elif mask.shape[-1] > data.mask_DC.shape[-1]: 
                mask = up_sample(data.mask_DC, stride, method=down_sample_method.replace("up","top").replace("down","bottom")) * mask
            elif mask.shape[-1] < data.mask_DC.shape[-1]:
                mask = down_sample(data.mask_DC, stride, method=down_sample_method.replace("up","top").replace("down","bottom")) * mask
            pred = model(data,mask) if training else model(data,mask)[-1]
        else:
            pred = model(data.LR)
    elif type(model).__name__ == "VisionTransformer":
        pred = model(data.LR)
    else:
        raise ValueError("Unknown model type")
    return pred

def mask_loss(pred, target, mask,loss_type = "l1"):
    if loss_type == "l1":
        loss = (F.l1_loss(pred,target, reduction="none") * mask.unsqueeze(1)).sum() / mask.sum()
    elif loss_type == "l2":
        loss = (F.mse_loss(pred,target, reduction="none") * mask.unsqueeze(1)).sum() / mask.sum()
    return loss

# %%
# trainer and evaluator
def trainer(model, optimizer, loader, device,gene_scale=10.,gene_loss_rate = 10.,down_sample_method="upleft",
            img_co_train=False,split_train=False,two_step_predict=False,train_on_img_only=False,img_grad=False,
            stride=2,gradient_loss_rate=0.):
    model = model.to(device)
    model.train() 
    losses, loss_gene1s, loss_gene1_grads,loss_gene2s,loss_img1_grads,loss_img2_grads, loss_img1s, loss_img2s = [], [], [], [], [], [], [], []
    if type(model).__name__ in ["RDN_Multi_Scale"]:
        for scale_factors in [[2],[4]]:            
            loss_gene, loss_gene_grad, loss_gene2, loss_img1_grad, loss_img2_grad, loss_img1, loss_img2 = 0., 0., 0., 0., 0., 0., 0.
            loss1, loss2,loss = 0., 0., 0.
            for data,data_img in loader:
                data = data.to(device)
                
                if not train_on_img_only:
                    optimizer.zero_grad()
                    ups1,downs1,downs2,ups2 = model(data.target/gene_scale, scale_factors)
                    for down1,up2,s in zip(downs1,ups2,scale_factors):
                        loss_gene1 = F.l1_loss(down1 * gene_scale,data.target, reduction="none")
                        loss_gene1 = (loss_gene1 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
                        loss_gene2 = F.l1_loss(up2 * gene_scale,data.target, reduction="none")
                        loss_gene2 = (loss_gene2 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
                        loss_gene = loss_gene1 + loss_gene2
                        if gradient_loss_rate:
                            loss_gene_grad += gradient_loss(up2,data.target/gene_scale,data.mask.unsqueeze(1)) + gradient_loss(down1,data.target/gene_scale,data.mask.unsqueeze(1))
                    for up1,down2,s in zip(ups1,downs2,scale_factors):
                        loss_gene1_ = F.l1_loss(down_sample(up1, s, method="topleft") * gene_scale,data.target, reduction="none")
                        loss_gene1_ = (loss_gene1 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
                        loss_gene2_ = F.l1_loss(down2 * gene_scale,down_sample(data.target, s, method="topleft"), reduction="none")
                        loss_gene2_ = (loss_gene2 * data.mask[:,::s,::s].unsqueeze(1)).sum() / data.mask[:,::s,::s].sum()
                        loss_gene += (loss_gene1_ + loss_gene2_) * 1. # lambda = 1.
                        
                        filter1 = nn.AvgPool2d(kernel_size=s*4,stride=s*4,padding=0)
                        filter2 = nn.AvgPool2d(kernel_size=4,stride=4,padding=0)
                        loss_gene_lfl1 = F.l1_loss(filter1(up1) * gene_scale,filter2(data.target), reduction="mean") * 1. # lambda2 = 1.
                        loss_gene_lfl2 = F.l1_loss(filter2(down2) * gene_scale,filter1(data.target), reduction="mean") * 1. # lambda2 = 1.
                        loss_gene += loss_gene_lfl1 + loss_gene_lfl2
                        
                    loss1 = loss_gene/gene_scale * gene_loss_rate + loss_gene_grad * gradient_loss_rate
                    loss1.backward()
                    optimizer.step()
                    loss = loss1.detach().item()
                
                    losses.append(loss)
                    loss_gene1s.append(loss_gene.item())
                    if gradient_loss_rate:
                        loss_gene1_grads.append(loss_gene_grad.item())
                 
                if img_co_train or train_on_img_only:
                    data_img = data_img.to(device)
                    optimizer.zero_grad()
                    
                    ups1_img,downs1_img,downs2_img,ups2_img = model(data_img.LR, scale_factors)
                    for down1,up2,s in zip(downs1_img,ups2_img,scale_factors):
                        loss_img = F.l1_loss(down1,data_img.target, reduction="mean") + F.l1_loss(up2,data_img.target, reduction="mean")
                        if gradient_loss_rate and img_grad:
                            loss_img2_grad = gradient_loss(up2,data_img.target) + gradient_loss(down1,data_img.target)
                    for up1,down2,s in zip(ups1_img,downs2_img,scale_factors):
                        loss_img += (F.l1_loss(down_sample(up1, s, method="topleft"),data_img.target, reduction="mean") + F.l1_loss(down2,down_sample(data_img.target, s, method="topleft"), reduction="mean")) * 1. # lambda = 1.
                        if gradient_loss_rate and img_grad:
                            loss_img2_grad += gradient_loss(up1,data_img.target) + gradient_loss(down2,data_img.target)
                        
                        filter1 = nn.AvgPool2d(kernel_size=s*4,stride=s*4,padding=0)
                        filter2 = nn.AvgPool2d(kernel_size=4,stride=4,padding=0)
                        loss_img_lfl1 = F.l1_loss(filter1(up1),filter2(data_img.target), reduction="mean") * 1. # lambda2 = 1.
                        loss_img_lfl2 = F.l1_loss(filter2(down2),filter1(data_img.target), reduction="mean") * 1. # lambda2 = 1.
                        loss_img += loss_img_lfl1 + loss_img_lfl2
                        
                    if two_step_predict:
                        assert False, "Not implemented"
                    loss2 = loss_img + loss_img2_grad * gradient_loss_rate
                    loss2.backward()
                    optimizer.step()
                    
                    if img_co_train:
                        loss += loss2.detach().item()
                        losses[-1] = loss
                    elif train_on_img_only:
                        loss = loss2.detach().item()
                        losses.append(loss)
                    
                    loss_img2s.append(loss_img.item())
                    if gradient_loss_rate and img_grad:
                        loss_img2_grads.append(loss_img2_grad.item())
            
    else:   
        for data,data_img in loader:
            optimizer.zero_grad()
            loss_gene1, loss_gene1_grad, loss_gene2, loss_img1_grad, loss_img2_grad, loss_img1, loss_img2 = 0., 0., 0., 0., 0., 0., 0.
            
            if "DCs" not in type(model).__name__:     # like ["RDN","RDN_M","RDNet","RDN_HAB","RDN_HABs","RDN_SPANs"]:
                if img_co_train or train_on_img_only:
                    data_img = data_img.to(device)
                    pred_img2 = get_pred(model,data_img,stride,down_sample_method,training = True)
                    loss_img2 = F.l1_loss(pred_img2,data_img.target)
                    if gradient_loss_rate and img_grad:
                        loss_img2_grad = gradient_loss(pred_img2,data_img.target)
                    if two_step_predict and not args.real_LR:
                        data_img.target = down_sample(data_img.target, stride, method=down_sample_method.replace("up","top").replace("down","bottom"))
                        data_img.LR = down_sample(data_img.target, stride, method=down_sample_method)
                        pred_img1 = get_pred(model,data_img,stride,down_sample_method,training = True)
                        loss_img1 = F.l1_loss(pred_img1,data_img.target)
                        if gradient_loss_rate and img_grad:
                            loss_img1_grad = gradient_loss(pred_img1,data_img.target)
                    if split_train and img_co_train:
                        loss_img = loss_img1 + loss_img2 + loss_img1_grad * gradient_loss_rate + loss_img2_grad * gradient_loss_rate
                        loss_img.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                if not train_on_img_only:
                    data = data.to(device)
                    pred_gene = get_pred(model,data,stride,down_sample_method,training = True)
                    loss_gene1 = mask_loss(pred_gene * gene_scale,data.target,data.mask,loss_type = "l1")
                    if gradient_loss_rate:
                        loss_gene1_grad = gradient_loss(pred_gene,data.target/gene_scale,data.mask.unsqueeze(1))
                    if two_step_predict:
                        data.LR = (up_sample(data.target, stride, method=down_sample_method) if not model_cls(type(model).__name__) else data.target.clone()) / gene_scale
                        pred_gene2 = get_pred(model,data,stride,down_sample_method,training = True)
                        loss_gene2 = mask_loss(down_sample(pred_gene2, stride, method=down_sample_method.replace("up","top").replace("down","bottom")) * gene_scale,data.target,data.mask,loss_type = "l1")
                    
                    if split_train and img_co_train:
                        loss_gene = loss_gene1/gene_scale * gene_loss_rate + loss_gene2/gene_scale * gene_loss_rate + loss_gene1_grad * gradient_loss_rate
                        loss_gene.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
            elif "DCs" in type(model).__name__:# like ["Unet_DCs","RDN_M_DCs","RDN_HAB_M_DCs","RDN_HABs_M_DCs","RDN_SPANs_M_DCs"]:
                if img_co_train or train_on_img_only:
                    data_img = data_img.to(device)
                    rs = get_pred(model,data_img,stride,down_sample_method,training = True)
                    loss_img2 = torch.sum(torch.stack([F.l1_loss(r,data_img.target) * (i+1)/len(rs) for i,r in enumerate(rs)]))
                    if gradient_loss_rate and img_grad:
                        loss_img2_grad = gradient_loss(rs[-1],data_img.target)
                    if two_step_predict and not args.real_LR:
                        data_img.target = down_sample(data_img.target, stride, method=down_sample_method.replace("up","top").replace("down","bottom"))
                        data_img.LR = down_sample(data_img.target, stride, method=down_sample_method)
                        rs = get_pred(model,data_img,stride,down_sample_method,training = True)
                        loss_img1 = torch.sum(torch.stack([F.l1_loss(r,data_img.target) * (i+1)/len(rs) for i,r in enumerate(rs)]))
                        if gradient_loss_rate and img_grad:
                            loss_img1_grad = gradient_loss(rs[-1],data_img.target)
                    if split_train and img_co_train:
                        loss_img = loss_img1 + loss_img2 + loss_img1_grad * gradient_loss_rate + loss_img2_grad * gradient_loss_rate
                        loss_img.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                if not train_on_img_only:
                    data = data.to(device)
                    rs = get_pred(model,data,stride,down_sample_method,training = True)
                    ls = [mask_loss(r*gene_scale,data.target,data.mask,loss_type = "l1") * (i+1)/len(rs) for i,r in enumerate(rs)]
                    loss_gene1 = torch.sum(torch.stack(ls))
                
                    if gradient_loss_rate:
                        loss_gene1_grad = gradient_loss(rs[-1],data.target/gene_scale,data.mask.unsqueeze(1))
                        
                    if two_step_predict:
                        data.LR = up_sample(data.target, stride, method=down_sample_method)
                        data.LR = data.LR/gene_scale
                        rs = get_pred(model,data,stride,down_sample_method,training = True)
                        ls = [mask_loss(down_sample(r, stride, method=down_sample_method.replace("up","top").replace("down","bottom")) * gene_scale,data.target,data.mask,loss_type = "l1") * (i+1)/len(rs) for i,r in enumerate(rs)]
                        loss_gene2 = torch.sum(torch.stack(ls))
                    if split_train and img_co_train:
                        loss_gene = loss_gene1/gene_scale * gene_loss_rate + loss_gene2/gene_scale * gene_loss_rate + loss_gene1_grad * gradient_loss_rate
                        loss_gene.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
            loss = loss_gene1/gene_scale * gene_loss_rate + loss_gene2/gene_scale * gene_loss_rate + loss_img1 + loss_img2 + loss_gene1_grad * gradient_loss_rate + loss_img1_grad * gradient_loss_rate + loss_img2_grad * gradient_loss_rate
            if not (split_train and img_co_train):
                loss.backward()
                optimizer.step()
            
            losses.append(loss.item())
            if isinstance(loss_gene1,torch.Tensor):
                loss_gene1s.append(loss_gene1.item())
            if gradient_loss_rate:
                loss_gene1_grads.append(loss_gene1_grad.item())
            if two_step_predict:
                if isinstance(loss_gene2,torch.Tensor):
                    loss_gene2s.append(loss_gene2.item())
            if img_co_train or train_on_img_only:
                loss_img2s.append(loss_img2.item())
                if gradient_loss_rate and img_grad:
                    loss_img2_grads.append(loss_img2_grad.item())
                if two_step_predict and not args.real_LR:
                    loss_img1s.append(loss_img1.item())
                    if gradient_loss_rate and img_grad:
                        loss_img1_grads.append(loss_img1_grad.item())
    return np.nanmean(losses), np.nanmean(loss_gene1s), np.nanmean(loss_gene1_grads),np.nanmean(loss_gene2s),np.nanmean(loss_img1s), np.nanmean(loss_img1_grads), np.nanmean(loss_img2s), np.nanmean(loss_img2_grads)

def evaluator(model, loader, device,gene_scale=10.,stride=2,down_sample_method="upleft"):
    model.to(device)
    model.eval()
    losses = []
    # loss_grads = []
    with torch.no_grad():
        for data,_ in loader:
            data = data.to(device)
            if type(model).__name__ in ["RDN_Multi_Scale"]:
                up1 = model.forward_up(data.LR,stride)
                loss = F.l1_loss(up1*gene_scale,data.target, reduction="none")
                loss = (loss * data.mask.unsqueeze(1)).sum() / data.mask.sum()
            else:
                pred = get_pred(model,data,stride,down_sample_method,training = False)
                loss = mask_loss(pred*gene_scale,data.target,data.mask,loss_type = "l1")
            losses.append(loss.item())
    return np.nanmean(losses)

# %%
# function to get metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,mean_absolute_error
#from sklearn.metrics.pairwise import cosine_similarity

def get_metrics(model,loader,device="cuda",sample_times=1,mask=False,compare_LR=False,compare_mean=False,stride=None,return_pred=False,gene_scale = 1.): 
    pccs,psnr_values,ssim_values,rmses,maes = [],[],[],[],[]
    pccs2,psnr_values2,ssim_values2,rmses2,maes2 = [],[],[],[],[]
    pccs3,psnr_values3,ssim_values3,rmses3,maes3 = [],[],[],[],[]

    y_pses = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data,_ in loader:
            data = data.to(device)
            patch_size = data.target.shape[-2]
            if model.__class__.__name__ in ["RDN_Multi_Scale"]:
                y_ps = model(data.target/gene_scale)[0][stride//2 - 1]
            elif model.__class__.__name__ == "UNet":
                y_ps = model.sample(None,data)
            elif "DCs" not in type(model).__name__:# like ["RDN","RDN_M","RDNet","RDN_HAB","RDN_HABs","RDN_SPANs"]:
                y_ps = model(data.LR)
            elif "DCs" in type(model).__name__: # like ["Unet_DCs","RDN_M_DCs","RDN_HAB_M_DCs","RDN_HABs_M_DCs","RDN_SPANs_M_DCs"]:
                assert stride is not None, "stride is required for DCs models"
                mask_ = torch.zeros_like(data.target).to(device)
                mask_[:,:,::stride,::stride] = 1.
                rs = model(data,mask_)
                y_ps = rs[-1]
    
            if mask:
                masks = repeat(data.mask,"b h w -> b c h w",c=data.target.shape[1])
                y_ps = y_ps * masks
                y_ps[y_ps<0] = 0
                
            y_ps = y_ps * gene_scale
            y_pses.append(y_ps)
            
            y1 = data.target.detach().cpu().numpy()
            y2 = y_ps.detach().cpu().numpy()
            
            if compare_LR:
                if data.LR.shape[-2] == data.target.shape[-2]:
                    y3 = data.LR.detach().cpu().numpy() *gene_scale # y LR
                else:
                    data.LR = F.interpolate(data.LR, scale_factor=data.target.shape[-2] // data.LR.shape[-2], mode='bicubic') * gene_scale
                    y3 = data.LR.detach().cpu().numpy() 
                
            if compare_mean:
                assert stride is not None, "stride is required for compare_mean"
                data.mean_value = get_average(data.target,stride = stride).to(device)
                y_4 = data.mean_value.detach().cpu().numpy() # y mean
            
            from scipy.stats import pearsonr
            if data.target.shape[1] == 1:
                t,t2,t3 = [],[],[]
                for i in range(y1.shape[0]):
                    t.append(pearsonr(y1[i].flatten(),y2[i].flatten())[0])
                    if compare_LR:
                        t2.append(pearsonr(y1[i].flatten(),y3[i].flatten())[0])
                    if compare_mean:
                        t3.append(pearsonr(y1[i].flatten(),y_4[i].flatten())[0])
                pccs.append(np.nanmean(t))
                if compare_LR:
                    pccs2.append(np.nanmean(t2))
                if compare_mean:
                    pccs3.append(np.nanmean(t3))
            else:
                t,t2,t3 = [],[],[]
                y1_ = rearrange(y1,"b c h w -> (b h w) c")
                y2_ = rearrange(y2,"b c h w -> (b h w) c")
                if compare_LR:
                    y3_ = rearrange(y3,"b c h w -> (b h w) c")
                for i in range(y1_.shape[0]):
                    t.append(pearsonr(y1_[i],y2_[i])[0])
                    if compare_LR:
                        t2.append(pearsonr(y1_[i],y3_[i])[0])
                    if compare_mean:
                        t3.append(pearsonr(y1_[i],y_4[i])[0])
                pccs.append(np.nanmean(t))
                if compare_LR:
                    pccs2.append(np.nanmean(t2))
                if compare_mean:
                    pccs3.append(np.nanmean(t3))
            
            psnr_values.append(psnr(y1, y2,data_range = data.target.max().item() - data.target.min().item()))
            ssim_values.append(ssim(y1.reshape(-1,patch_size,patch_size),y2.reshape(-1,patch_size,patch_size),data_range = max(y1.max(),y2.max()),channel_axis = 0))
            
            tem_l = F.mse_loss(y_ps, data.target, reduction='none')
            tem_l = tem_l * data.mask.unsqueeze(1)
            tem_l = tem_l.sum() / data.mask.sum()
            rmses.append(np.sqrt(tem_l.item()))
            
            tem_l = F.l1_loss(y_ps, data.target, reduction='none')
            tem_l = tem_l * data.mask.unsqueeze(1)
            tem_l = tem_l.sum() / data.mask.sum()
            maes.append(tem_l.item())
            
            if compare_LR:
                psnr_values2.append(psnr(y1, y3,data_range = max(data.target.max(),data.LR.max()).item()))
                ssim_values2.append(ssim(y1.reshape(-1,patch_size,patch_size),y3.reshape(-1,patch_size,patch_size),data_range = max(y1.max(),y3.max()),channel_axis = 0))
                
                tem_l = F.mse_loss(data.LR, data.target, reduction='none')
                tem_l = tem_l * data.mask.unsqueeze(1)
                tem_l = tem_l.sum() / data.mask.sum()
                rmses2.append(np.sqrt(tem_l.item()))
                
                tem_l = F.l1_loss(data.LR, data.target, reduction='none')
                tem_l = tem_l * data.mask.unsqueeze(1)
                tem_l = tem_l.sum() / data.mask.sum()
                maes2.append(tem_l.item())
            
            if compare_mean:
                psnr_values3.append(psnr(y1, y_4,data_range = max(data.target.max(),data.mean_value.max()).item()))
                ssim_values3.append(ssim(y1.reshape(-1,patch_size,patch_size),y_4.reshape(-1,patch_size,patch_size),data_range = max(y1.max(),y_4.max()),channel_axis = 0))

                tem_l = F.mse_loss(data.mean_value, data.target, reduction='none')
                tem_l = tem_l * data.mask.unsqueeze(1)
                tem_l = tem_l.sum() / data.mask.sum()
                rmses3.append(np.sqrt(tem_l.item()))
                
                tem_l = F.l1_loss(data.mean_value, data.target, reduction='none')
                tem_l = tem_l * data.mask.unsqueeze(1)
                tem_l = tem_l.sum() / data.mask.sum()
                maes3.append(tem_l.item())
                
    if compare_LR:
        m_ = pd.DataFrame({"PSNR":[np.nanmean(psnr_values),np.nanmean(psnr_values2)],
            "SSIM":[np.nanmean(ssim_values),np.nanmean(ssim_values2)],
            "RMSE":[np.nanmean(rmses),np.nanmean(rmses2)],
            "MAE":[np.nanmean(maes),np.nanmean(maes2)],
            "PCC":[np.nanmean(pccs),np.nanmean(pccs2)]},
            index=["Truth-Predict","Truth-LR"])
    else:
        m_ = pd.DataFrame({"PSNR":[np.nanmean(psnr_values)],"SSIM":[np.nanmean(ssim_values)],
                             "RMSE":[np.nanmean(rmses)],"MAE":[np.nanmean(maes)],
                             "PCC":[np.nanmean(pccs)]},
                            index=["Truth-Predict"])
    if compare_mean:
        m_mean = pd.DataFrame({"PSNR":[np.nanmean(psnr_values3)],
                            "SSIM":[np.nanmean(ssim_values3)],
                            "RMSE":[np.nanmean(rmses3)],
                            "MAE":[np.nanmean(maes3)],
                            "PCC":[np.nanmean(pccs3)]},
                            index=["Truth-Mean"])
        m_ = pd.concat([m_,m_mean],axis=0)
        
    if return_pred:
        return m_,y_pses
    return m_

# %%
model_note = f"_{args.model_note}" if args.model_note else ""
model_note = f"_gradient_loss{str(args.gradient_loss_rate)}{model_note}" if args.gradient_loss_rate else model_note
model_note = f"_img_grad{model_note}" if args.img_grad else model_note
model_note = f"_gene_loss_rate{args.gene_loss_rate}{model_note}" if args.gene_loss_rate not in [10,10.] else model_note
model_note = f"_gene_scale{args.gene_scale}{model_note}" if args.gene_scale not in [10,10.] else model_note
model_note = f"_drop_rate{args.drop_rate}{model_note}" if args.drop_rate not in [0,0.,None] else model_note
model_note = f"_train_on_img_only{model_note}" if args.train_on_img_only else model_note
model_note = f"_real_LR{model_note}" if args.real_LR else model_note
model_note = f"_downsample_{args.down_sample_method1}{model_note}" if args.down_sample_method1 != "upleft" else model_note
model_note = f"_cv_{args.cv}{model_note}" if args.cv != "upleft" else model_note

model_note_f = f"{args.model}{'_img_co_train' if args.img_co_train else ''}"
model_note_f = f"{model_note_f}{'_split_train' if args.split_train else ''}"
model_note_f = f"{model_note_f}{'_train_on_origin_size' if args.train_on_origin_size else ''}"
model_note_f = f"{model_note_f}{'_two_step_predict' if args.two_step_predict else ''}"


if "_DCs" in args.model:
    model_note = f"{model_note}_Clayers{model.Clayers}"
if "RDN" in args.model and args.model != "RDNet":
    model_note = f"{model_note}_G0{model.G0}_D{model.D}_C{model.C}_G{model.G}"

save_dir = f"./model_save/{model_note_f}_{args.train_dataset}_{args.marker}_{patch_size}x{patch_size}_stride{str(stride)}{model_note}/"
args.checkpoint_path = args.checkpoint_path if args.checkpoint_path else f'{save_dir}/last.pth'

print(f"Model: {args.model}")
print(f"Model note: {model_note}")
print(f"Image co-train: {args.img_co_train}")
print(f"Save dir: {save_dir}")
print(f"Model: {args.model}, Marker: {args.marker}, Patch size: {patch_size}x{patch_size}, Stride: {str(stride)}")
print(f"Train dataset: {args.train_dataset}, Test dataset: {args.test_dataset}, Metric dataset: {args.metric_dataset}")
print(f"Gene scale: {args.gene_scale}")
print(f"Train on origin size: {args.train_on_origin_size}")
print(f"Batch size: {batch_size}, Min mask rate: {min_mask_rate}, Drop rate: {drop_rate}")
print(f"Epochs: {args.epochs}, Change lr to: {args.change_lr_to}")
print(f"Checkpoint path: {args.checkpoint_path}")

# %%
# Baseline: bicubic, upleft, bilinear
if args.epochs>1:
    print(f"Baseline on test loss:")
    ds = Gene_Dataset(gene_exp_matrix,mask_whole,train_gs,down_sample_method=args.down_sample_method1,
                                           model=args.model, marker_idx=marker_index,stride=stride,gene_scale=args.gene_scale,patch_size=patch_size,
                                           training=False)

    l_bicubic,l_upleft,l_bilinear = [],[],[]
    for data,_ in ds:
        LR_bicubic = down_sample(data.target, stride, method="bicubic") 
        LR_upleft = down_sample(data.target, stride, method="upleft")
        LR_bilinear = down_sample(data.target, stride, method="bilinear")
        l1 = F.l1_loss(LR_upleft,data.target, reduction="none")
        l1 = (l1 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
        l_upleft.append(l1.item())
        l2 = F.l1_loss(LR_bicubic,data.target, reduction="none")
        l2 = (l2 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
        l_bicubic.append(l2.item())
        l3 = F.l1_loss(LR_bilinear,data.target, reduction="none")
        l3 = (l3 * data.mask.unsqueeze(1)).sum() / data.mask.sum()
        l_bilinear.append(l3.item())
    print(f"Upleft: {np.nanmean(l_upleft)} | Bicubic: {np.nanmean(l_bicubic)} | Bilinear: {np.nanmean(l_bilinear)}")

# %%
if args.epochs>0:
    train_losses,train_gene_losses,train_gene_grad_losses,train_gene2_losses, train_img1_losses,train_img2_losses,train_img1_grad_losses,train_img2_grad_losses = [],[],[],[],[],[],[],[]
    test_losses, lrs = [], []
    min_test_loss = 1000

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path,map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
        if os.path.dirname(args.checkpoint_path) == os.path.dirname(save_dir):
            train_losses = checkpoint["train_loss"] if checkpoint.get("train_loss") else []
            train_gene_losses = checkpoint["train_gene_loss"] if checkpoint.get("train_gene_loss") else []
            train_gene_grad_losses = checkpoint["train_gene_grad_loss"] if checkpoint.get("train_gene_grad_loss") else []
            train_gene2_losses = checkpoint["train_gene2_loss"] if checkpoint.get("train_gene2_loss") else []
            train_img1_losses = checkpoint["train_img1_loss"] if checkpoint.get("train_img1_loss") else []
            train_img2_losses = checkpoint["train_img2_loss"] if checkpoint.get("train_img2_loss") else []
            train_img1_grad_losses = checkpoint["train_img1_grad_loss"] if checkpoint.get("train_img1_grad_loss") else []
            train_img2_grad_losses = checkpoint["train_img2_grad_loss"] if checkpoint.get("train_img2_grad_loss") else []
            test_losses = checkpoint["test_loss"] if checkpoint.get("test_loss") else []
            lrs = checkpoint["lrs"] if checkpoint.get("lrs") else []
            epoch_from = checkpoint["epoch"]+1
            min_test_loss = min(test_losses) if test_losses else 1000
            print(f"Load model from {args.checkpoint_path}, start from epoch {epoch_from+1}")
        else:
            print(f"Load model from {args.checkpoint_path}")
            epoch_from = 0
    
    print(f'Save model to: {save_dir}')
    
    if save_dir:
        if os.path.exists(f'{save_dir}/min_loss.pth'):
            tem_path = f'{save_dir}/min_loss.pth'
            shutil.copy(tem_path,f'{save_dir}/min_loss_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')
        if os.path.exists(f'{save_dir}/last.pth'):    
            tem_path = f'{save_dir}/last.pth'
            shutil.copy(tem_path,f'{save_dir}/last_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')

    if args.change_lr_to is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.change_lr_to


    for epoch in range(epoch_from, epoch_from + args.epochs):
        train_loss, train_gene_loss, train_gene_grad_loss,train_gene2_loss, train_img1_loss, train_img1_grad_loss, train_img2_loss,  train_img2_grad_loss  = trainer(
            model, optimizer, train_loader, device,gene_scale=args.gene_scale,gene_loss_rate=args.gene_loss_rate,down_sample_method=args.down_sample_method1,
            img_co_train=args.img_co_train,split_train=args.split_train,two_step_predict=args.two_step_predict,train_on_img_only=args.train_on_img_only,img_grad=args.img_grad,
            stride=stride,gradient_loss_rate=args.gradient_loss_rate)
        test_loss = evaluator(model, test_loader, device,gene_scale=args.gene_scale,stride=stride,down_sample_method=args.down_sample_method1)
        train_losses.append(train_loss)
        train_gene_losses.append(train_gene_loss)
        train_gene_grad_losses.append(train_gene_grad_loss)
        train_gene2_losses.append(train_gene2_loss)
        train_img1_losses.append(train_img1_loss)
        train_img2_losses.append(train_img2_loss)
        train_img1_grad_losses.append(train_img1_grad_loss)
        train_img2_grad_losses.append(train_img2_grad_loss)
        test_losses.append(test_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        
        
        if test_loss < min_test_loss:
            checkpoint_path = f"{save_dir}/min_loss.pth"
            min_test_loss = test_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses,
                "train_gene_loss": train_gene_losses,
                "train_gene_grad_loss": train_gene_grad_losses,
                "train_gene2_loss": train_gene2_losses,
                "train_img1_loss": train_img1_losses,
                "train_img2_loss": train_img2_losses,
                "train_img1_grad_loss": train_img1_grad_losses,
                "train_img2_grad_loss": train_img2_grad_losses,
                "test_loss": test_losses,
                "epoch": epoch,
                "lrs": lrs,
                "save_path": save_dir,
                "checkpoint_path": checkpoint_path
            },checkpoint_path)
        
        if (epoch+1) % 5 == 0 or epoch == (epoch_from + args.epochs-1):
            checkpoint_path = f"{save_dir}/last.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses,
                "train_gene_loss": train_gene_losses,
                "train_gene_grad_loss": train_gene_grad_losses,
                "train_gene2_loss": train_gene2_losses,
                "train_img1_loss": train_img1_losses,
                "train_img2_loss": train_img2_losses,
                "train_img1_grad_loss": train_img1_grad_losses,
                "train_img2_grad_loss": train_img2_grad_losses,
                "test_loss": test_losses,
                "epoch": epoch,
                "lrs": lrs,
                "save_path": save_dir,
                "checkpoint_path": checkpoint_path
            },checkpoint_path)
        
        print(f"Epoch: {epoch+1}, Train gene loss: {train_gene_loss:.4f}, Train gene grad loss: {train_gene_grad_loss:.4f}, Train gene2 loss: {train_gene2_loss:.4f}, Train img1 loss: {train_img1_loss:.4f}, Train img1 grad loss: {train_img1_grad_loss:.4f}, Train img2 loss: {train_img2_loss:.4f}, Train img2 grad loss: {train_img2_grad_loss:.4f}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")        

    print(f"Save model to {save_dir}")

# %% [markdown]
# # Visualize the data

# %%
model.load_state_dict(torch.load(args.checkpoint_path,map_location='cpu')["model_state_dict"],strict=True)

# %%
metric_loader = DataLoader(Gene_Dataset(gene_exp_matrix,mask_whole,train_gs,down_sample_method=args.down_sample_method1,
                                           model=args.model, marker_idx=marker_index,stride=stride,gene_scale=args.gene_scale,patch_size=patch_size,
                                           training=False), 
                              batch_size=1, shuffle=False, num_workers=4)

# %%
r_ds = []
for d,_ in metric_loader:
    if d.upleft[0][0] % stride== x_ and d.upleft[0][1] %  stride == y_:
        r_ds.append(d)

# %%
pred_matrix = torch.zeros(row_max,col_max)
truth_matrix = torch.zeros(row_max,col_max) 
bicubic_matrix = torch.zeros(row_max,col_max) 
bilinear_matrix = torch.zeros(row_max,col_max) 

pred__ = torch.full((len(r_ds),row_max,col_max), float('nan'))
mask_count = torch.zeros(row_max,col_max) 
mask_whole = torch.zeros(row_max,col_max) 

model.eval()
with torch.no_grad():
    for i,d in enumerate(r_ds):
        #########################################################################################################
        # change this part to your model to get the prediction
        # data processing can be found on My_dataset class
        d = d.clone()
        p_bicubic = down_sample(d.target, stride, method="bicubic")
        p_bilinear = down_sample(d.target, stride, method="bilinear")
        
        truth_matrix[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] = d.target[0,0,:,:]
        
        bicubic_matrix[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] += p_bicubic[0,0,:,:]
        bilinear_matrix[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] += p_bilinear[0,0,:,:]
        
        d = d.to(device)
        # preediction can be found on elvaluater function
        if args.model in ["RDN_Multi_Scale"]:
            p = model.forward_up(d.LR,stride) * d.mask * args.gene_scale
        elif args.model in ["UNet","Unet"]:
            p = model(None,d) * d.mask * args.gene_scale 
        elif "DCs" not in args.model:# like ["RDN","RDN_M","RDNet","RDN_HAB_M","RDN_HAB","RDN_HABs","RDN_HABs_M","RDN_SPANs","RDN_SPANs_M"]:
            p = model(d.LR) * d.mask * args.gene_scale 
        elif "DCs" in args.model:# like ["Unet_DCs","RDN_M_DCs","RDN_HAB_M_DCs","RDN_HABs_M_DCs","RDN_SPANs_M_DCs"]:
            mask = torch.zeros_like(d.target).to(device)
            mask[:,:,::stride,::stride] = 1.
            rs = model(d,mask)
            p = rs[-1] * d.mask * args.gene_scale
        #########################################################################################################
        pred_matrix[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] += p.detach().cpu()[0,0,:,:] 
        pred__[i,d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] = p.detach().cpu()[0,0,:,:]
        
        mask_count[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] += 1
        mask_whole[d.upleft[0][0]:d.upleft[0][0]+patch_size,d.upleft[0][1]:d.upleft[0][1]+patch_size] = d.mask[0,:,:]

# %%
pred = torch.clamp(pred_matrix/mask_count,0,100).nan_to_num(0.) * mask_whole
truth = torch.clamp(truth_matrix,0,100).nan_to_num(0.) * mask_whole
bicubic_HR_from_patches = torch.clamp(bicubic_matrix/mask_count,0,100).nan_to_num(0.) * mask_whole
bilinear_HR_from_patches = torch.clamp(bilinear_matrix/mask_count,0,100).nan_to_num(0.) * mask_whole

bicubic_HR_from_global = torch.clamp(down_sample(truth.unsqueeze(0).unsqueeze(0), stride, method="bicubic")[0,0,:,:],0,100) * mask_whole
bilinear_HR_from_global = torch.clamp(down_sample(truth.unsqueeze(0).unsqueeze(0), stride, method="bilinear")[0,0,:,:],0,100) * mask_whole

upleft_LR = down_sample(truth.unsqueeze(0).unsqueeze(0), stride, method=args.cv)[0,0,:,:] * mask_whole

error1 = pred - truth
error2 = upleft_LR - truth
error3 = bicubic_HR_from_patches - truth
error4 = bicubic_HR_from_global - truth
error5 = bilinear_HR_from_patches - truth
error6 = bilinear_HR_from_global - truth

metric1 = (error1 * mask_whole).abs().sum()/mask_whole.sum()
metric2 = (error2 * mask_whole).abs().sum()/mask_whole.sum()
metric3 = (error3 * mask_whole).abs().sum()/mask_whole.sum()
metric4 = (error4 * mask_whole).abs().sum()/mask_whole.sum()
metric5 = (error5 * mask_whole).abs().sum()/mask_whole.sum()
metric6 = (error6 * mask_whole).abs().sum()/mask_whole.sum()

gradient_loss = GradientLoss2D().to("cpu")
grad1 = gradient_loss(pred.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))
grad2 = gradient_loss(upleft_LR.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))
grad3 = gradient_loss(bicubic_HR_from_patches.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))
grad4 = gradient_loss(bicubic_HR_from_global.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))
grad5 = gradient_loss(bilinear_HR_from_patches.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))
grad6 = gradient_loss(bilinear_HR_from_global.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),mask_whole.unsqueeze(0).unsqueeze(0))

print(f"{args.model}{' w/' if args.img_co_train else ' w/o'}  natural image co train {model_note}")
print(f"Pred-Truth:\t\t\t MAE: {metric1:.4f} \tGrad_MAE: {grad1:.4f}, \tupleft_LR-Truth:\t\t MAE: {metric2:.4f} \tGrad_MAE: {grad2:.4f}")
print(f"Bicubic from patches-Truth:\t MAE: {metric3:.4f} \tGrad_MAE: {grad3:.4f}, \tBicubic from global-Truth:\t MAE: {metric4:.4f} \tGrad_MAE: {grad4:.4f}")
print(f'Bilinear from patches-Truth:\t MAE: {metric5:.4f} \tGrad_MAE: {grad5:.4f}, \tBilinear from global-Truth:\t MAE: {metric6:.4f} \tGrad_MAE: {grad6:.4f}')

pred__mean = pred__.nanmean(dim=0)
pred__std = ((pred__ - pred__mean).square().nansum(0) / (mask_count-1)).sqrt()# Not a perfect methods, but it's ok

# %%
m = torch.zeros_like(truth)
m[x_::stride,y_::stride] = 1
m = m*mask_whole

pred_replace = pred * (1-m) + truth * m
bicubic_HR_from_patches_replace = bicubic_HR_from_patches * (1-m) + truth * m
bilinear_HR_from_patches_replace = bilinear_HR_from_patches * (1-m) + truth * m

bicubic_HR_from_global_replace = bicubic_HR_from_global * (1-m) + truth * m
bilinear_HR_from_global_replace = bilinear_HR_from_global * (1-m) + truth * m

error_replace1 = pred_replace - truth
error_replace2 = upleft_LR - truth
error_replace3 = bicubic_HR_from_patches_replace - truth
error_replace4 = bicubic_HR_from_global_replace - truth
error_replace5 = bilinear_HR_from_patches_replace - truth
error_replace6 = bilinear_HR_from_global_replace - truth

metric_replace1 = (error_replace1 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()
metric_replace2 = (error_replace2 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()
metric_replace3 = (error_replace3 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()
metric_replace4 = (error_replace4 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()
metric_replace5 = (error_replace5 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()
metric_replace6 = (error_replace6 * (mask_whole - m)).abs().sum()/(mask_whole - m).sum()

grad_replace1 = gradient_loss(pred_replace.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))
grad_replace2 = gradient_loss(upleft_LR.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))
grad_replace3 = gradient_loss(bicubic_HR_from_patches_replace.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))
grad_replace4 = gradient_loss(bicubic_HR_from_global_replace.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))
grad_replace5 = gradient_loss(bilinear_HR_from_patches_replace.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))
grad_replace6 = gradient_loss(bilinear_HR_from_global_replace.unsqueeze(0).unsqueeze(0),truth.unsqueeze(0).unsqueeze(0),(mask_whole - m).unsqueeze(0).unsqueeze(0))

replace_error1 = (pred_replace - pred).abs().sum()/m.sum()
replace_error2 = (upleft_LR - upleft_LR).abs().sum()/m.sum()
replace_error3 = (bicubic_HR_from_patches_replace - bicubic_HR_from_patches).abs().sum()/m.sum()
replace_error4 = (bicubic_HR_from_global_replace - bicubic_HR_from_global).abs().sum()/m.sum()
replace_error5 = (bilinear_HR_from_patches_replace - bilinear_HR_from_patches).abs().sum()/m.sum()
replace_error6 = (bilinear_HR_from_global_replace - bilinear_HR_from_global).abs().sum()/m.sum()

print(f"{args.model}{' w/' if args.img_co_train else ' w/o'}  natural image co train {model_note} Keep known spots")
print(f"Pred_replace-Truth:\t\t\t MAE: {metric_replace1:.4f} \tGrad_MAE: {grad_replace1:.4f} \tReplace error: {replace_error1:.4f}, \tLR-Truth:\t\t\t\t MAE: {metric_replace2:.4f} \tGrad_MAE: {grad_replace2:.4f} \tReplace error: {replace_error2:.4f}")
print(f"Bicubic from patches replace-Truth:\t MAE: {metric_replace3:.4f} \tGrad_MAE: {grad_replace3:.4f} \tReplace error: {replace_error3:.4f}, \tBicubic from global replace-Truth:\t MAE: {metric_replace4:.4f} \tGrad_MAE: {grad_replace4:.4f} \tReplace error: {replace_error4:.4f}")
print(f'Bilinear from patches replace-Truth:\t MAE: {metric_replace5:.4f} \tGrad_MAE: {grad_replace5:.4f} \tReplace error: {replace_error5:.4f}, \tBilinear from global replace-Truth:\t MAE: {metric_replace6:.4f} \tGrad_MAE: {grad_replace6:.4f} \tReplace error: {replace_error6:.4f}')


# %%
vmax = max(pred.max(),truth.max(),bicubic_HR_from_patches.max(),bicubic_HR_from_global.max(),bilinear_HR_from_patches.max(),bilinear_HR_from_global.max(),
            pred_replace.max(),bicubic_HR_from_patches_replace.max(),bicubic_HR_from_global_replace.max(),bilinear_HR_from_patches_replace.max(),bilinear_HR_from_global_replace.max())

error_max = max(error1.abs().max(),error2.abs().max(),error3.abs().max(),error4.abs().max(),error5.abs().max(),error6.abs().max(),
                error_replace1.abs().max(),error_replace2.abs().max(),error_replace3.abs().max(),error_replace4.abs().max(),error_replace5.abs().max(),error_replace6.abs().max())

error_min = -error_max
vmin = 0

print(f"Vmax of Gene: {vmax}, Vmax of error: {error_max}")
#vmax = 6.6
#vmin = 0
#error_max = 6

torch.save({"pred":pred,"truth":truth,"mask_whole":mask_whole},
           f"./prediction/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}.pth")
print(f"Save prediction to f'./prediction/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}.pth'")

print("Plotting...,saving to './image_save/' ")

# %%
def plt_(plot_data,
         cmap = "Blues",
         title = "",
        vmax = None,
        vmin = None,
        title_fontsize = 16,
        colors_bar = None,
        save_path = None):
    
    if vmax is None:
        vmax = abs(plot_data).max()
    if vmin is None:
        vmin = -vmax if plot_data.min() < 0 else 0
        
    plt.imshow(plot_data,cmap=cmap,vmax=vmax,vmin=vmin)
    if colors_bar:
        plt.colorbar()
    plt.axis("off")
    plt.title(title,fontsize=title_fontsize)
    
    if save_path:
        plt.savefig(save_path,dpi=300,transparent=True)
        if "pdf" in save_path:
            plt.savefig(save_path.replace("pdf","png"),dpi=300,transparent=True)
        if "png" in save_path:
            plt.savefig(save_path.replace("png","pdf"),dpi=300,transparent=True)
    plt.show()

from matplotlib.colors import LinearSegmentedColormap
# 定义颜色
colors = ["blue", "white","red"]  # Blue to white to red
n_bins = 100  # 这决定了颜色的渐变步数

# 创建颜色映射对象
error_camp = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_bins)

blues_camp = LinearSegmentedColormap.from_list("custom_cmap", ["white", "#4682b4"], N=n_bins)

# %%
plt_(truth,
     cmap="Blues",
     title='',# "Truth",
     colors_bar=False,vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_truth.png")

# %%
plt_(upleft_LR,
     cmap="Blues",
     title='',#f"Input LR",
     colors_bar=False,vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_upleftLR.png")

plt_(error2,
     cmap=error_camp,
     title='',# "Error: LR - Truth",
     colors_bar=False,vmax=error_max,vmin=-error_max,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_upleftLR_error.png")

# %%
plt_(bicubic_HR_from_patches,
     title="",# "Bicubic HR from patches",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_patches.png")

plt_(error3,
     title="",# "Error: Bicubic HR from patches - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_patches_error.png")


plt_(bicubic_HR_from_patches_replace,
     title="",# "Bicubic HR from patches",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_patches_replace.png")

plt_(error_replace3,
     title="",# "Error: Bicubic HR from patches - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_patches_replace_error.png")

# %%
plt_(bicubic_HR_from_global,
     title="",# "Bicubic HR from global",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_global.png")

plt_(error4,
     title="",# "Error: Bicubic HR from global - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_global_error.png")

plt_(bicubic_HR_from_global_replace,
     title="",# "Bicubic HR from global",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_global_replace.png")

plt_(error_replace4,
     title="",# "Error: Bicubic HR from global - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bicubic_HR_from_global_replace_error.png")

# %%
plt_(bilinear_HR_from_patches,
     title="",# "Bi-linear HR from patches",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_biliner_HR_from_patches.png")

plt_(error5,
     title="",# "Error: Bi-linear HR from patches - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_patches_error.png")

plt_(bilinear_HR_from_patches_replace,
     title="",# "Bi-linear HR from patches",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_patches_replace.png")

plt_(error_replace5,
     title="",# "Error: Bi-linear HR from patches - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_patches_replace_error.png")

# %%
plt_(bilinear_HR_from_global,
     title="",# "Bi-linear HR from global",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_global.png")

plt_(error6,
     title="",# "Error: Bi-linear HR from global - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_global_error.png")

plt_(bilinear_HR_from_global_replace,
     title="",# "Bi-linear HR from global",
     colors_bar=False,
     vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_global_replace.png")

plt_(error_replace6,
     title="",# "Error: Bi-linear HR from global - Truth",
     cmap=error_camp,
     vmax=error_max,
     vmin=-error_max,
     colors_bar=False,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_bilinear_HR_from_global_replace_error.png")

# %%
plt_(pred,
     title="",# {args.model} {'w/' if args.img_co_train else 'w/o'} real image co-train Prediction HR",
     colors_bar=False,vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}_Pred.png")

plt_(error1,
     cmap=error_camp,
     title= "",# f"{args.model} {'w/' if args.img_co_train else 'w/o'} real image co-train Error Prediction HR - Truth",
     colors_bar=False,vmax=error_max,vmin=-error_max,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}_Pred_error.png")

plt_(pred_replace,
     title="",# {args.model} {'w/' if args.img_co_train else 'w/o'} real image co-train Prediction HR",
     colors_bar=False,vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}_Pred_replace.png")

plt_(error_replace1,
     cmap=error_camp,
     title= "",# f"{args.model} {'w/' if args.img_co_train else 'w/o'} real image co-train Error Prediction HR - Truth",
     colors_bar=False,vmax=error_max,vmin=-error_max,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_{args.model}{'_natural_image_co_train' if args.img_co_train else ''}{'_train_on_origin_size' if args.train_on_origin_size else ''}{model_note}_Pred_replace_error.png")


# %%
plt_(np.zeros_like(truth),
     cmap="Blues",
     title='',
     colors_bar=True,vmax=vmax,vmin=vmin,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_gene_colors_bar.png")

plt_(np.zeros_like(truth),
     cmap=error_camp,
     title='',
     colors_bar=True,vmax=error_max,vmin=-error_max,
     save_path=f"./image_save/{train_dataset_names[0]}_{args.marker}_error_colors_bar.png")

# %%
print("Ploting finished.")

