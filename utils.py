import os
import sys
from datetime import datetime
import shutil
import concurrent.futures

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from PIL import Image

import h5py
import yaml

import scanpy as sc
from einops import rearrange,reduce,repeat
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torchvision import transforms

# Function to load data
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to save data
def save_data(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
# load and save yaml
def load_yaml(yaml_file):
    with open(yaml_file,"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def save_yaml(config,yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f)
        
# Function to read spatial data
def read_spatial_data(spatial_count_path,loc_path=None):
    adata = sc.read(spatial_count_path,delimiter='\t')
    adata.var_names_make_unique("--")
    if loc_path:
        loc = pd.read_csv(loc_path,sep="\t")
        adata.obsm['spatial'] = loc.values
    if not isinstance(adata.X, np.ndarray):
        try:
            adata.X = adata.X.toarray()
        except:
            raise ValueError(f"adata.X type is {type(adata.X)}, cannot convert to numpy array")
    adata.layers["raw"] = adata.X.copy()
    if "ENSG" == adata.var_names[0][:4]:
        from pyensembl.shell import collect_all_installed_ensembl_releases
        from pyensembl import EnsemblRelease

        # EnsemblRelease().download() run this in python, or 
        # pyensembl install --release 111 --species human, run this in command line
        # pyensembl list, run this in command line to check installed species and releases
        data = EnsemblRelease(release=111, species='homo_sapiens')
        ns = []
        for i in adata.var_names:
            try:
                ns.append(data.gene_name_of_gene_id(i))
            except:
                ns.append(i)
        adata.var_names = ns
    return adata

# Function to reset 'Spatial Transcriptomics' Tech data's index
def reset_ST_index(adata):
    index = adata.obs.index
    index = [i.split("x") for i in index]
    index = np.array(index).astype(float).round(0).astype(int)
    adata.obs['array_row'] = index[:, 0]
    adata.obs['array_col'] = index[:, 1]
    adata.obs.index = [f'{index[i, 0]:03}x{index[i, 1]:03}' for i in range(len(index))]
    return adata

# Function to reset 'Xenium' Tech data's columns, because some datasets do not have array_row and array_col in adata.obs
def reset_Xenium_columns(adata):
    unique_col = np.unique(adata.obsm['spatial'][:,0]).tolist()
    unique_row = np.unique(adata.obsm['spatial'][:,1]).tolist()
    cors = [(unique_col.index(cor[0]),unique_row.index(cor[1])) for cor in adata.obsm['spatial']]
    adata = adata.copy()
    adata.obs['array_col'] = np.array(cors)[:,0]
    adata.obs['array_row'] = np.array(cors)[:,1]
    return adata

# Function to reset 'Xenium' Tech data's columns, because some datasets do not have array_row and array_col in adata.obs
def reset_Visium_columns(adata):
    adata = adata.copy()
    loc = adata.obs[['array_row',"array_col"]].values
    adata.obs['array_row_raw'] = loc[:,0]
    adata.obs['array_col_raw'] = loc[:,1]
    for i in loc:
        if i[0] % 2 == 0:
            i[1] = i[1] / 2
        elif i[1] % 2 == 1:
            i[1] = (i[1] - 1) / 2
    adata.obs['array_row'] = loc[:,0]
    adata.obs['array_col'] = loc[:,1]
    return adata

# Function for load patch image
def load_patches(adata,patches_path):
    with h5py.File(patches_path, 'r') as file:
        # print(file.keys())
        coords = file['coords'][:]
        barcode = file['barcode'][:]
        patch_images = file['img'][:]
    barcodes = [str(i[0]).split("'")[1] for i in barcode.tolist()]
    selected_index = [i in barcodes for i in adata.obs_names]
    adata = adata[selected_index]
    selected_index2 = [i for i in range(len(barcodes)) if barcodes[i] in adata.obs_names]
    barcodes = [barcodes[i] for i in selected_index2]

    adata.obsm["patch_image"] = patch_images[selected_index2]
    adata.obs["barcode"] = barcodes
    adata.obsm["coord"] = coords[selected_index2]
    assert sum(barcodes != adata.obs_names) == 0, "barcode not match"
    return adata

# Function to plot spatial data
def qc_plot(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    axs[0].set_ylabel("Number of cells")
    axs[0].set_xlabel("Total counts for each cell")
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[1])
    axs[1].set_ylabel("Number of cells")
    axs[1].set_xlabel("Number of genes expressed by each cell")

    p = [0.01,0.05,0.1,0.15,0.2,0.8,0.85,0.9,0.95,0.99]
    columns = [f"{int(100*i)}%" for i in p]
    df = pd.DataFrame([np.quantile(adata.obs["total_counts"], p),
                        np.quantile(adata.obs["n_genes_by_counts"], p)],
                    columns=columns,
                    index=["total_counts", "n_genes_by_counts"])
    print(df)
    return df

# Function to filter cells
def filter_cells(adata,max_mt=None,min_counts=None):
    if max_mt:
        if "mt" not in adata.var:
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata = adata[adata.obs["pct_counts_mt"] < max_mt].copy()
    
    if min_counts:
        if isinstance(min_counts,(float,int)):
            sc.pp.filter_cells(adata, min_counts=min_counts)
        
    return adata

# Function to normalize and log transform data
def log_normalize(adata):
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    return adata

# Function to filter genes
def filter_genes(adata,n_top_genes,min_cells=None,min_counts=None):
    # use percentage to filter
    if min_cells:
        if isinstance(min_cells, float):
            min_cells = int(min_cells * adata.n_obs)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
    if min_counts:
        if isinstance(min_counts, float):
            min_counts = int(min_counts * adata.n_obs)
        sc.pp.filter_cells(adata, min_counts=min_counts)
        
    log_normalize(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    
    adata = adata[:, adata.var.highly_variable].copy()
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    return adata

# single thread version
def get_regions(adata, row_min, row_max, col_min, col_max, size=16,min_mask_rate=0.5):
    idxs = []
    masks = []
    uplefts = []
    # regions = []         
    
    array_row = adata.obs["array_row"]
    array_col = adata.obs["array_col"]
    ls = np.arange(adata.obsm["patch_image"].shape[0])
    obs_dict = {}
    [obs_dict.update({f"{r:04}x{c:04}":l}) for r,c,l in zip(array_row,array_col,ls)]
    for row_idx,col_idx in np.ndindex(row_max-row_min+1,col_max-col_min+1):
        row = row_min + row_idx
        col = col_min + col_idx

        print(f"Processing region {row}-{col}/{row_max}-{col_max}",end="\r")
        idx = np.zeros((size*size,)) - 1
        mask = np.zeros((size, size))
        # region = np.zeros((size*224, 16*224,3))
        
        for i,j in np.ndindex(size,size):
            row_idx = row + i
            col_idx = col + j
            idx_bool = obs_dict.get(f"{row_idx:04}x{col_idx:04}",None)
            if idx_bool is not None:
                idx[i*size+j] = idx_bool
                mask[i, j] = 1
                
        if mask.sum() >= size*size*min_mask_rate:
            idxs.append(idx)
            masks.append(mask)
            uplefts.append((row, col))
    
    idxs = torch.tensor(np.array(idxs),dtype=torch.long)
    masks = torch.tensor(np.array(masks),dtype=torch.long)
    uplefts = torch.tensor(np.array(uplefts),dtype=torch.long)
    return idxs,masks,uplefts #regions, patch_imgs

# original version
def get_regions_(adata, row_min, row_max, col_min, col_max, size=16,min_mask_rate=0.5):
    gene_exps = []
    # patch_imgs = []
    patch_embs = []
    masks = []
    uplefts = []
    # regions = []            
    for row_idx,col_idx in np.ndindex(row_max-row_min+1,col_max-col_min+1):
        row = row_min + row_idx
        col = col_min + col_idx

        print(f"Processing region {row}-{col}/{row_max}-{col_max}",end="\r")
        gene_exp = np.zeros((size, size, adata.X.shape[1]))
        patch_emb = np.zeros((size, size, adata.obsm["patch_emb"].shape[1]))
        # patch_img = np.zeros((size, size, 224, 224, 3))
        mask = np.zeros((size, size))
        # region = np.zeros((size*224, 16*224,3))
        
        for i,j in np.ndindex(size,size):
            row_idx = row + i
            col_idx = col + j
            idx_bool = (adata.obs["array_row"] == row_idx) & (adata.obs["array_col"] == col_idx)
            if sum(idx_bool) > 0:
                # patch_img[i, j, :, :, :] = adata.obsm["patch_image"][idx_bool,:,:,:]
                gene_exp[i, j, :] = adata.X[idx_bool, :]
                patch_emb[i, j, :] = adata.obsm["patch_emb"][idx_bool,:]
                # region[i*224:(i+1)*224, j*224:(j+1)*224,:] = adata.obsm["patch_img"][idx_bool,:,:,:]
                mask[i, j] = 1
        if mask.sum() >= size*size*min_mask_rate:
            # patch_imgs.append(patch_img)
            gene_exps.append(gene_exp)
            patch_embs.append(patch_emb)
            # regions.append(region)
            masks.append(mask)
            uplefts.append((row, col))
    gene_exps = torch.tensor(np.array(gene_exps),dtype=torch.float32)
    # patch_imgs = torch.tensor(np.array(patch_imgs),dtype=torch.float32)
    patch_embs = torch.tensor(np.array(patch_embs),dtype=torch.float32)
    # regions = torch.tensor(np.array(regions),dtype=torch.float32)
    masks = torch.tensor(np.array(masks),dtype=torch.long)
    uplefts = torch.tensor(np.array(uplefts),dtype=torch.long)
    return gene_exps, patch_embs, masks,uplefts #regions, patch_imgs

# Function to get calculated average
def get_average(xx,stride=4):
    x = xx.clone()
    shape = x.shape
    if len(x.shape) == 3:
        x = rearrange(x, 'c h w -> 1 c h w')
    x = torch.nn.functional.avg_pool2d(x, kernel_size=stride, stride=stride)
    b, c, h, w = x.shape
    expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
    for i in range(h):
        for j in range(w):
            expanded_tensor[:, :, stride * i:stride * i + stride, stride * j:stride * j + stride] = x[:, :, i:i + 1, j:j + 1]
    return expanded_tensor if len(shape) == 4 else expanded_tensor[0]

def down_sample(xx,stride=4,method="upleft"):
    x = xx.clone()
    shape = x.shape
    if len(x.shape) == 3:
        x = rearrange(x, 'c h w -> 1 c h w')
    if method == "topleft":
        expanded_tensor = x[:,:,::stride,::stride]
    elif method == "topright":
        expanded_tensor = x[:,:,::stride,1::stride]
    elif method == "bottomleft":
        expanded_tensor = x[:,:,1::stride,::stride]
    elif method == "bottomright":
        expanded_tensor = x[:,:,1::stride,1::stride]
    elif method == "random":
        x = x[:,:,::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h, w)
        row_idx = torch.randint(0,stride,(h,))
        col_idx = torch.randint(0,stride,(w,))
        for i,j in np.ndindex(h,w):
            expanded_tensor[:, :, i, j] = xx[:, :, stride * i + row_idx[i], stride * j + col_idx[j]]
    elif method == "upleft":
        x = x[:,:,::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
        for i,j in np.ndindex(stride,stride):
            expanded_tensor[:, :, i::stride, j::stride] = x
    elif method == "upright":
        x = x[:,:,::stride,1::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
        for i,j in np.ndindex(stride,stride):
            expanded_tensor[:, :, i::stride, j::stride] = x
    elif method == "downleft":
        x = x[:,:,1::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
        for i,j in np.ndindex(stride,stride):
            expanded_tensor[:, :, i::stride, j::stride] = x
    elif method == "downright":
        x = x[:,:,1::stride,1::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
        for i,j in np.ndindex(stride,stride):
            expanded_tensor[:, :, i::stride, j::stride] = x
    elif method == "zeros":
        x = x[:,:,::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = torch.zeros(b, c, h * stride, w * stride)
        expanded_tensor[:, :, ::stride, ::stride] = x
    elif method in ["nearest","bilinear","bicubic","trilinear"]:
        x = x[:,:,::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = F.interpolate(x, size=(h*stride, w*stride), mode=method, align_corners=False)
    elif method in ["nearest_replace","bilinear_replace","bicubic_replace","trilinear_replace"]:
        x = x[:,:,::stride,::stride]
        b, c, h, w = x.shape
        expanded_tensor = F.interpolate(x, size=(h*stride, w*stride), mode=method.replace("_replace",""), align_corners=False)
        expanded_tensor[:,:,::stride,::stride] = x
    else:
        raise ValueError("method should be 'upleft' or 'nearest' or 'bilinear' or 'bicubic' or 'trilinear'")
    return expanded_tensor.to(xx.device) if len(shape) == 4 else expanded_tensor[0].to(xx.device)

def up_sample(xx,stride=2,method="upleft"):
    x = xx.clone()
    if len(x.shape) == 3:
        x = rearrange(x, 'c h w -> 1 c h w')
    if method in ["upleft","upright","downleft","downright"]:
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        for i,j in np.ndindex(stride,stride):
            expanded_tensor[:,:,i::stride,j::stride] = x
    elif method == "topright":
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        expanded_tensor[:,:,::stride,::stride] = x
    elif method == "topleft":
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        expanded_tensor[:,:,::stride,1::stride] = x
    elif method == "bottomleft":
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        expanded_tensor[:,:,1::stride,0::stride] = x
    elif method == "downright":
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        expanded_tensor[:,:,1::stride,1::stride] = x 
    elif method == "zeros":
        expanded_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*stride, x.shape[3]*stride)
        expanded_tensor[:,:,::stride,::stride] = x
    elif method in ["nearest","bilinear","bicubic","trilinear"]:
        expanded_tensor = F.interpolate(x, size=(x.shape[2]*stride, x.shape[3]*stride), mode=method, align_corners=False)
    elif method in ["nearest_replace","bilinear_replace","bicubic_replace","trilinear_replace"]:
        expanded_tensor = F.interpolate(x, size=(x.shape[2]*stride, x.shape[3]*stride), mode=method.replace("_replace",""), align_corners=False)
        expanded_tensor[:,:,::stride,::stride] = x
    else:
        raise ValueError("method should be 'upleft' or 'nearest' or 'bilinear' or 'bicubic' or 'trilinear'")

    return expanded_tensor.to(xx.device) if len(xx.shape) == 4 else expanded_tensor[0].to(xx.device)


# convert region to Data object
def get_Data(gene_exp,patch_emb,mask,idx,upleft,dataset_idx=None,stride=4,down_sample_method="upleft",patch_img=None,marker_idx=None):
    gene_exp = rearrange(gene_exp, 'h w c -> c h w')
    patch_img = rearrange(patch_img, 'h w c -> c h w') if patch_img is not None else torch.tensor([])
    patch_emb = rearrange(patch_emb, 'h w c -> c h w') 
    # x = rearrange(gene_exp, 'c h w -> (h w) c')
    # edge_index
    loc = torch.argwhere(mask == 1).to(torch.float32)
    dists = torch.cdist(loc,loc)
    dists.fill_diagonal_(float('inf'))
    edge_index = torch.nonzero(dists < 1.01).T.long()
    # mean value
    # mean_value = get_average(gene_exp,stride)
    #residual
    # LR = down_sample(gene_exp,stride,down_sample_method)
    # residual = gene_exp - LR
    # residual_mean = get_average(residual,stride)
    
    marker_idx = torch.tensor(marker_idx) if marker_idx is not None else torch.tensor([])
    dataset_idx = dataset_idx if dataset_idx is not None else torch.tensor([])
    # unsqueeze to add batch dimension
    data = Data(x= None, # x,                            # use for GNN so that no need to unsqueeze
                edge_index=edge_index,          # use for GNN so that no need to unsqueeze
                mask=mask.unsqueeze(0),
                gene_exp=gene_exp.unsqueeze(0),   
                mean_value=None,# mean_value.unsqueeze(0),
                patch_img=patch_img.unsqueeze(0),
                patch_emb=patch_emb.unsqueeze(0),
                residual=None, # residual.unsqueeze(0),
                residual_mean = None,# residual_mean.unsqueeze(0),
                LR=None,# LR.unsqueeze(0),
                target=None,# gene_exp.unsqueeze(0),
                idx=idx.unsqueeze(0),
                upleft=upleft.unsqueeze(0),
                dataset_idx = dataset_idx.unsqueeze(0),
                marker_idx=marker_idx.unsqueeze(0))
    return data

def sample_data_simple(datasets,lengths,ratio= 0.5,min_mask_rate=0.5):
    train_dataset, test_dataset = [],[]
    for i in range(len(lengths)-1):
        train_dataset.extend(datasets[lengths[i]:lengths[i] + int((lengths[i+1]-lengths[i])*ratio)])
        test_dataset.extend(datasets[lengths[i] + int((lengths[i+1]-lengths[i])*ratio):lengths[i+1]])
    train_dataset = [i for i in train_dataset if i.mask.sum() >= i.mask.numel()*min_mask_rate]
    test_dataset = [i for i in test_dataset if i.mask.sum() >= i.mask.numel()*min_mask_rate]
    return train_dataset, test_dataset

def sample_data(idxses,maskses,datasets,ratio= 0.5,min_mask_rate=0.5):
    train_idxses, test_idxses = [],[]
    train_maskses, test_maskses = [],[]
    train_datasets, test_datasets = [],[]
    for idxs,masks,dataset in zip(idxses,maskses,datasets):
        ii = masks.sum(dim=(1,2)) >= min_mask_rate*masks[0].numel()
        idxses = idxs[ii]
        maskses = masks[ii]
        dataset = [dataset[i] for i in range(len(dataset)) if ii[i]]
        
        if ratio <= 1.:
            train_idxs, test_idxs = idxses[:int(len(idxses)*ratio)], idxses[int(len(idxses)*ratio):]
            train_masks, test_masks = maskses[:int(len(idxses)*ratio)], maskses[int(len(idxses)*ratio):]
            train_dataset, test_dataset = dataset[:int(len(idxses)*ratio)], dataset[int(len(idxses)*ratio):]
        elif ratio > 1.:
            ii2 = [i for i in range(len(dataset)) if dataset[i].upleft[0,0] % int(ratio) == 0 and dataset[i].upleft[0,1] % int(ratio) == 0]
            ii3 = [i for i in range(len(dataset)) if i not in ii2]
            train_idxs, test_idxs = idxses[ii2], idxses[ii3]
            train_masks, test_masks = maskses[ii2], maskses[ii3]
            train_dataset, test_dataset = [dataset[i] for i in ii2], [dataset[i] for i in ii3]

        train_idxses.append(train_idxs)
        test_idxses.append(test_idxs)
        train_maskses.append(train_masks)
        test_maskses.append(test_masks)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        
    return train_idxses,train_maskses,train_datasets, test_idxses, test_maskses,test_datasets

# Dataset class
class My_Dataset(Dataset):
    def __init__(self, datasets,adatas,idxses,residual=False,stride=4,model_type=None,down_sample_method="upleft",
                 patch_emb_method="Conch",train_on_each_gene=False,target_level=1,transform=False,marker_idx=None,drop_rate=0.):
        self.datasets = datasets
        self.adatas = adatas
        self.idxses = idxses
        self.lengths = [0]
        for i in range(len(idxses)):
            self.lengths.append(self.lengths[-1]+idxses[i].shape[0])
        self.residual = residual
        self.stride = stride
        self.model_type = model_type
        self.down_sample_method = down_sample_method
        self.patch_emb_method = patch_emb_method
        self.train_on_each_gene = train_on_each_gene
        if marker_idx is not None:
            if type(marker_idx) != list:
                marker_idx = [marker_idx]
        self.marker_idx = torch.tensor(marker_idx) if marker_idx is not None else None
        self.means = [torch.from_numpy(i.X).mean(0).view(1,-1,1,1).to(torch.float32) for i in adatas]
        self.stds = [torch.from_numpy(i.X).std(0).view(1,-1,1,1).to(torch.float32) for i in adatas]
        self.transform = transform
        
        if self.patch_emb_method == "Trainable":
            self.img_means = [(torch.from_numpy(i.obsm["patch_image"])/255).mean(dim=(0,1,2)).cpu().numpy() for i in adatas]
            self.img_stds = [(torch.from_numpy(i.obsm["patch_image"])/255).std(dim=(0,1,2)).cpu().numpy() for i in adatas]
            self.img_transform = [transforms.Compose([
                #transforms.Resize(368),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(10),
                #transforms.CenterCrop(224),
                #transforms.ColorJitter(
                #    brightness=0.2,  # 亮度变化范围，0.5表示亮度在 [0.5, 1.5] 之间变化
                #    contrast=0.2,    # 对比度变化范围
                #    saturation=0.2,  # 饱和度变化范围
                #    hue=0.1          # 色调变化范围，0.2 表示在 [-0.2, 0.2] 之间变化
                #),
                transforms.ToTensor(),
                #transforms.Normalize(mean=self.img_means[i],std=self.img_stds[i]),
                ]) for i in range(len(adatas))]
        self.target_level = target_level
        self.drop_rate = drop_rate
    
    def __len__(self):
        return self.lengths[-1]
    
    def __getitem__(self, idx):
        for i in range(len(self.lengths)):
            if idx < self.lengths[i+1]:
                break
        idx = idx - self.lengths[i]
        
        index = self.idxses[i][idx]
        data = self.datasets[i][idx].clone()
        adata = self.adatas[i]

        if self.marker_idx is not None:
            marker_idx = self.marker_idx
        if self.train_on_each_gene:
            if len(self.marker_idx) != 1:
                marker_idx = torch.randint(0, data.gene_exp.shape[1], (1,))    
            
        data.target = data.gene_exp[:,marker_idx,:,:]
        data.marker_idx = data.marker_idx[:,marker_idx]
        
        if self.transform:
            data.target = (data.gene_exp - self.means[i]) / self.stds[i]
        
        if self.patch_emb_method == "Trainable":
            data.patch_img = torch.zeros((1,len(index),3,224,224))
            for i in range(len(index)):
                img = Image.fromarray(adata.obsm["patch_image"][index[i]])
                data.patch_img[0][i] = self.img_transform[i](img)
            if self.stride == 1:
                data.patch_img = data.patch_img.squeeze(0)
            
            # data.patch_img = torch.from_numpy(adata.obsm["patch_image"][index]/255).permute(0,3,1,2).to(torch.float32)
                
        if self.residual:
            mean_value = get_average(data.target,self.stride)
            data.target = data.target - mean_value
          
        if self.model_type != "RDN":
            if self.target_level == 1:
                data.LR = down_sample(data.target,self.stride,self.down_sample_method)
            elif self.target_level != 1:
                data.target = data.target[:,:,::self.target_level,::self.target_level]
                data.mask = data.mask[:,::self.target_level,::self.target_level]
                data.patch_emb = data.patch_emb[:,:,::self.target_level,::self.target_level]
                data.LR = down_sample(data.target,self.target_level,self.down_sample_method)

        elif self.model_type == "RDN":
            if self.target_level == 1:
                if self.down_sample_method == "random":
                    data.LR = down_sample(data.target,self.stride,self.down_sample_method)
                else: 
                    data.LR = data.target[:,:,::self.stride,::self.stride]
            elif self.target_level != 1:
                data.target = data.target[:,:,::self.target_level,::self.target_level]
                data.mask = data.mask[:,::self.target_level,::self.target_level]
                data.patch_emb = data.patch_emb[:,:,::self.target_level,::self.target_level]
                if self.down_sample_method == "random":
                    data.LR = down_sample(data.target,self.target_level,self.down_sample_method)
                else:
                    data.LR = data.target[:,:,::self.target_level,::self.target_level]
        
        m = torch.rand(data.LR.shape) > self.drop_rate if self.drop_rate > 0. else torch.ones(data.LR.shape)
        data.LR = data.LR * m if self.drop_rate > 0. else data.LR

        
        ###############################
        data.x = None
        data.edge_index = None
        data.gene_exp = None
        data.residual = None
        data.residual_mean = None
        data.mean_value = None
        ###############################
        return data

# trainer and evaluator
def trainer(model, optimizer, loader, device):
    model.to(device)
    model.train() 
    losses = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if type(model).__name__ == "UNet":
            loss = F.l1_loss(model(None,data),data.target)
        elif type(model).__name__ == "Diffusion":
            loss = model(data.target,data,data.mask)
        elif type(model).__name__ == "RDN":
            loss = F.l1_loss(model(data.LR),data.target)
        elif type(model).__name__ == "VAE":
            recon_batch, mu, logvar = model(data.target)
            loss = model.loss(data.target, recon_batch, mu, logvar)
        elif type(model).__name__ == "AE":
            loss = model.loss(data.target,model(data.target))
        elif type(model).__name__ == "RDNetImageEncoder":
            loss = model.loss(model(data.patch_img),data.target.squeeze(-1).squeeze(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.nanmean(losses)

def evaluator(model, loader, device):
    model.to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if type(model).__name__ == "UNet":
                loss = F.l1_loss(model(None,data),data.target)
            elif type(model).__name__ == "Diffusion":
                loss = model(data.target,data,data.mask)
            elif type(model).__name__ == "RDN":
                loss = F.l1_loss(model(data.LR),data.target)
            elif type(model).__name__ == "VAE":
                recon_batch, mu, logvar = model(data.target)
                loss = model.loss(data.target, recon_batch, mu, logvar)
            elif type(model).__name__ == "AE":
                loss = model.loss(data.target,model(data.target))
            elif type(model).__name__ == "RDNetImageEncoder":
                loss = model.loss(model(data.patch_img),data.target.squeeze(-1).squeeze(-1))
            losses.append(loss.item())
    return np.nanmean(losses)

# training loop
def training(model,optimizer,scheduler,
            device,config,epochs,
          train_loader,test_loader,metric_loader=None,
          change_lr_to=None,
          save_dir="",checkpoint_path = None,
          model_note:str="",
          min_loss = 100000,
          num_report = 10,
          max_metric = None,
          min_train_loss = 100000):
    
    metric_loader = metric_loader if metric_loader and max_metric else test_loader if max_metric else None
    if save_dir:
        if not checkpoint_path:
            if model_note:
                save_dir = f'{save_dir}/{type(model).__name__}_{model_note}_{datetime.now().strftime("%m-%d_%H:%M:%S")}/'
            else:
                save_dir = f'{save_dir}/{type(model).__name__}_{datetime.now().strftime("%m-%d_%H:%M:%S")}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    train_losses,test_losses,test_metrics,lrs = [],[],[],[]
    try:
        epoch_from = config['model_param']["model_training_parameters"]["epoch_from"]
    except:
        epoch_from = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path,map_location="cpu")
        if os.path.normcase(os.path.realpath(os.path.dirname(checkpoint_path))) != os.path.normcase(os.path.realpath(checkpoint['save_path'])):
            reset_checkpoint_path(checkpoint_path)
            checkpoint = torch.load(checkpoint_path,map_location="cpu")
        try:
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Warning: Model not loaded strictly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Optimizer not loaded")
        epoch_from = checkpoint['epoch']
        train_losses = checkpoint['train_losses'].tolist()
        test_losses = checkpoint['test_losses'].tolist()
        lrs = checkpoint['lrs'].tolist() if checkpoint.get('lrs',None) is not None else []
        test_metrics = checkpoint['test_metrics'] if checkpoint.get('test_metrics',False) and max_metric else []
        max_metric = np.nanmax(test_metrics) if (isinstance(test_metrics,list) and len(test_metrics)) and max_metric else max_metric if max_metric else None
        min_loss = np.nanmin(test_losses)
        min_loss = min_loss if min_loss else 100000
        min_train_loss = np.nanmin(train_losses)
        save_dir = os.path.dirname(checkpoint_path) + "/" if save_dir else ""
        

        print(f"Load model from {checkpoint_path}, start from epoch {epoch_from+1}")
        if save_dir:
            if os.path.exists(f'{save_dir}/min_loss.pth'):
                tem_path = f'{save_dir}/min_loss.pth'
                shutil.copy(tem_path,f'{save_dir}/min_loss_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')
            if os.path.exists(f'{save_dir}/last.pth'):    
                tem_path = f'{save_dir}/last.pth'
                shutil.copy(tem_path,f'{save_dir}/last_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')
            if os.path.exists(f'{save_dir}/best_metric.pth'):    
                tem_path = f'{save_dir}/best_metric.pth'
                shutil.copy(tem_path,f'{save_dir}/best_metric_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')
            if os.path.exists(f'{save_dir}/train_min_loss.pth'):    
                tem_path = f'{save_dir}/train_min_loss.pth'
                shutil.copy(tem_path,f'{save_dir}/train_min_loss_{torch.load(tem_path,map_location="cpu")["epoch"]}.pth')
    if save_dir:
        print(f"Start training {datetime.now().strftime('%m-%d_%H:%M:%S')}, save model to {save_dir}")
    if change_lr_to:
        for param_group in optimizer.param_groups:
            param_group['lr'] = change_lr_to
        
    for epoch in range(epoch_from+1,epoch_from+epochs+1):
        train_loss = trainer(model,optimizer,train_loader,device)
        test_loss = evaluator(model,test_loader,device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        if max_metric:
            # Do not support diffusion model, and only support one metric
            m = get_metrics(model,metric_loader,device)
            if isinstance(m,pd.DataFrame):
                m_v = m.iloc[0,0]
            else:
                m_v = m
            test_metrics.append(m_v)
            print(f"Epoch {epoch} ")
            print(m)
            if max_metric < m_v:
                max_metric = m_v
                save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note,"best_metric",save_dir,test_metrics,lrs)
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note,"train_min_loss",save_dir,test_metrics,lrs) 
        if test_loss < min_loss:
            min_loss = test_loss
            save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note,"min_loss",save_dir,test_metrics,lrs)
        
        if (epoch % num_report) == 0 or (epoch == epoch_from+1):
            print(f"Epoch {epoch} Train Loss: {train_loss} Test Loss: {test_loss} LR: {optimizer.param_groups[0]['lr']}")
        if (epoch % max(5,num_report)) == 0:
            save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note,"last",save_dir,test_metrics,lrs)
        
        if scheduler:
            if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
                
    print(f"Epoch {epoch} Train Loss: {train_loss} Test Loss: {test_loss} LR: {optimizer.param_groups[0]['lr']}")
    save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note,"last",save_dir,test_metrics,lrs)
    if save_dir:
        print(f"Training finished {datetime.now().strftime('%m-%d_%H:%M:%S')}, save model to {save_dir}")
    return train_losses,test_losses,save_dir
    
# save model
def save_checkpoint(model,optimizer,epoch,train_losses,test_losses,config,model_note:str,
                    checkpoint_name,save_dir,test_metrics = None,lrs = None,
                    verbose=False):
    checkpoint = {
        'model': type(model).__name__,
        "name": checkpoint_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
        'train_losses': torch.tensor(train_losses),
        'test_losses': torch.tensor(test_losses),
        'config' : config,
        'model_note' : model_note,
        "save_path" : save_dir,
        'test_metrics': test_metrics,
        'lrs': torch.tensor(lrs) if lrs else None
    }
    if save_dir:
        save_path = f'{save_dir}/{checkpoint_name}.pth'
        
        torch.save(checkpoint, save_path)
        save_yaml(config,f"{save_dir}/config.yaml")
        if verbose:
            print(f"Save model as {save_path}")
    
    return checkpoint

# reset checkpoint path when copy to new folder
def reset_checkpoint_path(checkpoint_path,path_only=False):
    tt = torch.load(checkpoint_path,map_location="cpu")
    tt['save_dir'] = os.path.dirname(checkpoint_path) + "/"
    tt['save_path'] = os.path.dirname(checkpoint_path) + "/"
    if not path_only:
        tt['epoch'] = 1
        tt["train_losses"] = torch.tensor([10.])
        tt["test_losses"] = torch.tensor([10.])
        tt["best_metric"] = [0]
    torch.save(tt,checkpoint_path)
    print(f"Reset checkpoint path to {os.path.normcase(os.path.realpath(os.path.dirname(checkpoint_path)))}")


# plot losses
def plot_losses(train_losses,test_losses,epoch_from,save_dir):
    x = np.arange(epoch_from,len(train_losses))
    plt.plot(x,train_losses[epoch_from:],label="Train Loss")
    plt.plot(x,test_losses[epoch_from:],label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/losses.png")
    plt.show()
    
# evaluate model
## PCC
def pearson_correlation(y_true, y_pred):    
    assert y_true.shape == y_pred.shape, "Shape of y_true and y_pred must be the same"
    
    mean_true = torch.mean(y_true, dim=-1, keepdim=True)
    mean_pred = torch.mean(y_pred, dim=-1, keepdim=True)
    
    y_true_diff = y_true - mean_true
    y_pred_diff = y_pred - mean_pred
    
    numerator = torch.sum(y_true_diff * y_pred_diff, dim=-1)
    denominator = torch.sqrt(torch.sum(y_true_diff ** 2, dim=-1) * torch.sum(y_pred_diff ** 2, dim=-1))
    r = numerator / denominator
    r = r[~torch.isnan(r)]
    return r.detach().mean().cpu().numpy().tolist()

## Cosine Similarity
def cosine_similarity(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
    return torch.nn.functional.cosine_similarity(y_true, y_pred, dim=-1).detach().mean().cpu().numpy().tolist()

## RMSE
def rmse(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2,dim = -1)).detach().mean().cpu().numpy().tolist()

## MAE
def mae(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
    return torch.mean(torch.abs(y_true - y_pred),dim=-1).detach().mean().cpu().numpy().tolist()

## metrics
def evaluate_model(model,loader,device,compare_mean_truth=False,diff_sample_times=1,sample_from_mean = False,time_step=500,clip_denoised = False,residual = False):
    model.eval()
    pccs, pccs2, pccs3 = [],[],[]
    cosines, cosines2, cosines3 = [],[],[]
    rmses, rmses2, rmses3 = [],[],[]
    maes, maes2, maes3 = [],[],[]
    stds = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if diff_sample_times>1:
                y_pred_all = torch.stack([model.sample(data.target.shape,data,clip_denoised) for _ in range(diff_sample_times)],dim=0)
                y_pred = y_pred_all.mean(dim=0)
                mask = repeat(data.mask,"b h w -> b c h w",c=y_pred.shape[1])
                y_pred = y_pred * mask

            elif diff_sample_times == 1:
                y_pred = model.sample(data.target.shape,data,clip_denoised).reshape(len(data),-1)
                mask = repeat(data.mask,"b h w -> b c h w",c=y_pred.shape[1])
                y_pred = y_pred * mask

            if residual:
                y_pred += data.mean_value
            
            y_pred = rearrange(y_pred,"b c h w->b c (h w)")
            y_true = rearrange(data.target,"b c h w->b c (h w)")
            pccs2.append(pearson_correlation(y_true,y_pred))
            cosines2.append(cosine_similarity(y_true,y_pred))
            rmses2.append(rmse(y_true,y_pred))
            maes2.append(mae(y_true,y_pred))
            if diff_sample_times>1:
                stds.append(y_pred_all.std().mean().cpu().numpy().tolist())
        if compare_mean_truth:
            y_mean = rearrange(data.mean_value,"b c h w->b c (h w)")
            y_target = rearrange(data.target,"b c h w->b c (h w)")
            pccs3.append(pearson_correlation(y_target,y_mean))
            cosines3.append(cosine_similarity(y_target,y_mean))
            rmses3.append(rmse(y_target,y_mean))
            maes3.append(mae(y_target,y_mean))
    tem = {"PCC":[np.nanmean(pccs2)],
            "Cosine":[np.mean(cosines2)],
            "RMSE":[np.mean(rmses2)],
            "MAE":[np.mean(maes2)],
            "STD":[np.mean(stds) if stds else None]}
    if compare_mean_truth:
        tem["PCC"].append(np.nanmean(pccs3))
        tem["Cosine"].append(np.mean(cosines3))
        tem["RMSE"].append(np.mean(rmses3))
        tem["MAE"].append(np.mean(maes3))
        tem["STD"].append(None)
        return pd.DataFrame(tem,index=["Mask-Truth-pred","Mask-Truth-mean"])
    return pd.DataFrame(tem,index=["All-Truth-pred"])


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
        for data in loader:
            data = data.to(device)
            patch_size = data.target.shape[-2]
            if model.__class__.__name__ == "Diffusion":
                if sample_times == 1:
                    y_ps = model.sample(data.target.shape,data)
                if sample_times > 1:
                    y_ps = torch.stack([model.sample(data.target.shape,data) for _ in range(sample_times)],dim=0).mean(dim=0)
            elif model.__class__.__name__ == "UNet":
                y_ps = model.sample(None,data)
            elif model.__class__.__name__ == "RDN":
                y_ps = model.sample(data.LR)
            elif model.__class__.__name__ == "VAE":
                y_ps,_,_ = model(data.target)
            elif model.__class__.__name__ == "AE":
                y_ps = model(data.target)
            elif model.__class__.__name__ == "RDNetImageEncoder":
                y_ps = model(data.patch_img)
            if mask:
                masks = repeat(data.mask,"b h w -> b c h w",c=data.target.shape[1])
                y_ps = y_ps * masks
                y_ps[y_ps<0] = 0
            
            y_ps = gene_scale * y_ps
            y_pses.append(y_ps)
            
            y1 = data.target.detach().cpu().numpy()
            if model.__class__.__name__ == "RDNetImageEncoder":
                y_ps = y_ps.unsqueeze(-1).unsqueeze(-1)
            y2 = y_ps.detach().cpu().numpy()
            
            if compare_LR:
                if data.LR.shape[-2] == data.target.shape[-2]:
                    y3 = data.LR.detach().cpu().numpy()*gene_scale # y LR
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