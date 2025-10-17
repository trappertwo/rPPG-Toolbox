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
    print("Testing uses Physnet pretrained model!")
    return model


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and normalize by its standard deviation."""
  
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    
    for j in range(diffnormalized_len):
      diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
          data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


class ImageDataSet(Dataset):
    """Dataset for SwinIR model"""
    
    def __init__(self, frames, window_size = 7):
        self.frames = frames
        self.window_size = window_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = frame.astype(np.float32)/ 255
        frame = frame.transpose(2, 0, 1)  # HWC-RGB to CHW-RGB
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        frame = torch.from_numpy(frame).float().to(device)
        # pad input image to be a multiple of window_size
        _, h_old, w_old = frame.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        frame = torch.cat([frame, torch.flip(frame, [1])], 1)[:, :h_old + h_pad, :]
        frame = torch.cat([frame, torch.flip(frame, [2])], 2)[:, :, :w_old + w_pad]
        return frame


class SwinIR(nn.Module):
    def __init__(self, swinir_model_path, window_size=7, img_size=126, batch_size=128, freeze=True):
        super(SwinIR, self).__init__()
        self.swinir_model = load_swinir_model(swinir_model_path, window_size=window_size, img_size=img_size)
        self.window_size = window_size
        self.img_size = img_size
        self.batch_size = batch_size
      
        # Freeze parameters of the model
        if freeze:
            for param in self.swinir_model.parameters():
                param.requires_grad = False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.swinir_model.to(device)

    def forward(self, frames):
        height, width, channel = frames[0].shape
        #print(frames.shape)
        image_ds = ImageDataSet(frames, self.window_size)
        image_dl = DataLoader(image_ds, batch_size=self.batch_size, shuffle=False)
      
        restored_frames = []
        for batch in image_dl:
            restored = self.swinir_model(batch)
            for i in range(restored.shape[0]):
                output = restored[i]
                output = output[..., :height, :width]
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = output.transpose(1, 2, 0)  # CHW-RGB to HWC-RGB
                    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                    restored_frames.append(output)
            print("Restored batch")
            #media.show_image(output)
        return np.array(restored_frames)


class SwinPhys(nn.Module):
    """Hybrid model combining SwinIR and PhysNet models"""
    
    def __init__(self, swinir_model_path, restore=True, physnet_model_path="", window_size=7, img_size=126, num_frames=128, freeze_swinir=True, freeze_physnet=True):
        super(SwinPhys, self).__init__()
        self.swinir_model = SwinIR(swinir_model_path, window_size=window_size, img_size=img_size, batch_size=num_frames, freeze=freeze_swinir)
        self.window_size = window_size
        self.img_size = img_size
        self.restore = restore
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Load pretrained physnet
        if physnet_model_path != "":
            self.physnet_model = load_physnet_model(physnet_model_path, num_frames=num_frames)
            if freeze_physnet:
                for param in self.physnet_model.parameters():
                param.requires_grad = False
            self.physnet_model.to(self.device)
        else:
            self.physnet_model = PhysNet_padding_Encoder_Decoder_MAX(
                frames=num_frames).to(self.device)  # [3, T, 128,128]

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape

        print("Input shape (CNWH): " + x.shape)
        restored_frames = x
        # Assume batch size of 1
        if self.restore:
            frames = x.squeeze().float().cpu().numpy()   # C, N, W, H
            frames = frames.transpose(1, 2, 3, 0)   # N, W, H, C
            restored_frames = self.swinir_model(frames) # # N, W, H, C
            print("Restored shape (NWHC): " + restored_frames.shape)
        restored_frames = diff_normalize_data(restored_frames) # N, W, H, C
        # Transpose to get data in the form C, N, W, H
        restored_frames = restored_frames.transpose(3, 0, 1, 2)
        print("After diff normalization (CNWH): " + restored_frames.shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        restored_frames = torch.from_numpy(restored_frames).float().unsqueeze(0).to(device) # batch_size, C, N, W, H
        return self.physnet_model(restored_frames)
