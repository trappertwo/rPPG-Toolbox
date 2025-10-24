""" SwinPhys
This is a fine-tuned hybrid model that combines the SwinIR model for jpeg compression with PhysNet.

Original SwinIR paper: SwinIR: Image Restoration Using Swin Transformer
Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte
github repo: https://github.com/JingyunLiang/SwinIR
"""

import torch
import torch.nn as nn
import os
import numpy as np

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from torch.utils.data import DataLoader, Dataset
from models.network_swinir import SwinIR as net
from torch.utils.data import TensorDataset


def load_swinir_model(model_path, window_size=7, img_size=126):
    """Loads the pretrained SwinIR model"""
    
    # set up model
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        raise ValueError(f'model {model_path} does not exist.')
    model = net(upscale=1, in_chans=3, img_size=img_size, window_size=window_size,
                img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def load_physnet_model(model_path, num_frames):
    """Loads a pretrained PhysNet model"""
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        raise ValueError(f'model {model_path} does not exist.')
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=num_frames)
    model.load_state_dict(torch.load(model_path))
    print("Using Physnet pretrained model!")
    return model


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = torch.zeros((diffnormalized_len, h, w, c), dtype=torch.float32)
    diffnormalized_data_padding = torch.zeros((1, h, w, c), dtype=torch.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / torch.std(diffnormalized_data)
    diffnormalized_data = torch.cat(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[torch.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


class ImageDataSet(Dataset):
    """Dataset for SwinIR model"""
    
    def __init__(self, frames, window_size = 7):
        """Initialize the Dataset with a tensor with the shape NWHC"""
        self.frames = frames
        self.window_size = window_size

    def __len__(self):
      return self.frames.shape[0]

    def __getitem__(self, idx):
        return self.frames[idx]  # C, W, H


class SwinIR(nn.Module):
    def __init__(self, swinir_model_path, normalize=True, window_size=7, img_size=126, batch_size=128, freeze=True):
        super(SwinIR, self).__init__()
        self.swinir_model = load_swinir_model(swinir_model_path, window_size=window_size, img_size=img_size)
        self.window_size = window_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.normalize = normalize
      
        # Freeze parameters of the model
        for param in self.swinir_model.parameters():
            param.requires_grad = False
        # Unfreeze the last RSTB block and the final conv layers
        if not freeze:
            # Unfreeze the last RSTB block
            #for layer in self.swinir_model.layers[-1:]:
            #    for param in layer.parameters():
            #        param.requires_grad = True
            # Unfreeze the last two convolutional layers
            for param in self.swinir_model.conv_after_body.parameters():
                param.requires_grad = True
            for param in self.swinir_model.conv_last.parameters():
                param.requires_grad = True
            print("Unfreezing layers from SwinIR")
                
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.swinir_model.to(device)

    def pad_frames(self, frames):
        """
        Args:
        frames (torch.Tensor): The input tensor of shape (N, C, W, H).
        window_size (int): The required factor for the padded dimensions.
        
        Returns:
        torch.Tensor: The padded tensor with shape (N, C, W_padded, H_padded).
        """    
        _, _, h_old, w_old = frames.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        if w_pad == 0 and h_pad == 0:
          return frames
        frames = torch.cat([frames, torch.flip(frames, [2])], 2)[:, :, :h_old + h_pad, :]
        frames = torch.cat([frames, torch.flip(frames, [3])], 3)[:, :, :, :w_old + w_pad]
        return frames
    
    def forward(self, frames):
        _, _, w_orig, h_orig = frames.shape # N, C, W, H
        if self.normalize:
            frames = frames.float() / 255.0
        frames = self.pad_frames(frames)

        image_ds = ImageDataSet(frames, self.window_size)
        image_dl = DataLoader(image_ds, batch_size=self.batch_size, shuffle=False)
      
        for i, batch in enumerate(image_dl):
            restored = self.swinir_model(batch)
            # Crop the padded area back to the original size (H_orig, W_orig)
            output_cropped = restored[:, :, :h_orig, :w_orig]
            # Clamp, scale to [0, 255], round, and convert to integer type (uint8 tensor)
            # .clamp_(0, 1) is in-place and ensures output is valid [0, 1] range
            if self.normalize:
                output = (output_cropped.clamp_(0, 1) * 255.0).round().to(torch.uint8)
            else:
                output = output_cropped.clamp_(0, 1)
            # assume we have a single batch
            return output


class SwinPhys(nn.Module):
    """Hybrid model combining SwinIR and PhysNet models"""
    
    def __init__(self, swinir_model_path, restore=True, physnet_model_path="", diff_normalize=True, window_size=7, img_size=126, num_frames=128, freeze_swinir=True, freeze_physnet=True):
        super(SwinPhys, self).__init__()
        self.swinir_model = SwinIR(swinir_model_path, normalize=diff_normalize, window_size=window_size, img_size=img_size, batch_size=num_frames, freeze=freeze_swinir)
        self.window_size = window_size
        self.img_size = img_size
        self.restore = restore
        self.diff_normalize = diff_normalize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Load pretrained physnet
        if physnet_model_path != "":
            self.physnet_model = load_physnet_model(physnet_model_path, num_frames=num_frames)
            if freeze_physnet:
                for param in self.physnet_model.parameters():
                    param.requires_grad = False
                print("Physnet is frozen")
            else:
                print("Unfreezing Physnet")
            self.physnet_model.to(self.device)
        else:
            self.physnet_model = PhysNet_padding_Encoder_Decoder_MAX(
                frames=num_frames).to(self.device)  # [3, T, 128,128]

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape   # NCDWH
        #print(f"Input shape: {x.shape}")
        restored_frames = x
        # Assume batch size of 1
        if self.restore:
            frames = x.squeeze().float()   # CDWH
            frames = frames.permute(1, 0, 2, 3)   # DCWH
            frames = self.swinir_model(frames)  # DCWH
            #print(f"After SwinIR DCWH: {frames.shape}")
            if self.diff_normalize:
                frames = frames.permute(0, 2, 3, 1)   # DWHC
                frames = diff_normalize_data(frames) # DWHC
                frames = frames.permute(3, 0, 1, 2)   # CDWH
            else:
                frames = frames.permute(1, 0, 2, 3)  # CDWH
            restored_frames = frames.float().unsqueeze(0)  # NCDWH
        return self.physnet_model(restored_frames)
