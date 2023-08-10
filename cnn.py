import numpy as np

def conv2d(inputs, weights, bias, padding, stride):
    C, H, W = inputs.shape
    o, c, h, w = weights.shape
    padded_inputs = np.pad(inputs, (0,0), (padding, padding), (padding, padding))
    h_out = (H - h + 2 * padding) // stride + 1
    w_out = (W - w + 2 * padding) // stride + 1
    outputs = np.zeros((o, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            sliced_inputs = padded_inputs[:, i*stride:i*stride+h, j*stride:j*stride+w]
            outputs[:, i, j] = np.sum(sliced_inputs * weights, axis=(1,2,3)) + bias
    return outputs

def maxpool2d(inputs, kernel_size, stride, padding):
    C, H, W = inputs.shape
    h, w = kernel_size, kernel_size
    padded_inputs = np.pad(inputs, (0,0), (padding, padding), (padding, padding))
    h_out = (H - kernel_size + 2 * padding) // stride + 1
    w_out = (W - kernel_size + 2 * padding) // stride + 1
    outputs = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            sliced_inputs = padded_inputs[:, i*stride:i*stride+h, j*stride:j*stride+w]
            outputs[:, i,j] = np.max(sliced_inputs, axis=(1,2))
    return outputs

def batchnorm(inputs, gamma, beta, eps=1e-6):
    N,C,H,W = inputs.shape
    mean = np.mean(inputs, axis=(0,2,3), keepdims=True)
    var = np.var(inputs, axis=(0,2,3), keepdims=True)
    normed_inputs = (inputs - mean) / np.sqrt(var + eps)
    return normed_inputs * gamma + beta

def layernorm(inputs, gamma, beta, eps=1e-6):
    N,C,H,W = inputs.shape
    mean = np.mean(inputs, axis=(1), keepdims=True)
    var = np.var(inputs, axis=(1), keepdims=True)
    normed_inputs = (inputs - mean) / np.sqrt(var + eps)
    return normed_inputs * gamma + beta
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.proj_q = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.proj_k = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.proj_v = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.proj_out = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)
        
    def forward(self, x):
        n, hw, c = x.shape
        query = self.proj_q(x).view(n, hw, self.num_heads, self.head_dim).transpose(1,2) #n, 8, hw, 32
        key = self.proj_k(x).view(n, hw, self.num_heads, self.head_dim).transpose(1,2) #n, 8, hw, 32
        value = self.proj_v(x).view(n, hw, self.num_heads, self.head_dim).transpose(1,2) #n, 8, hw, 32
        attn = torch.matmul(query, key.transpose(2,3)) / math.sqrt(self.head_dim) # n,8,hw,hw
        attn = F.softmax(attn, dim=-1)
        outputs = torch.matmul(attn, value).transpose(1,2).reshape(n, hw, c)
        outputs = self.proj_out(outputs)
        return outputs
        
    