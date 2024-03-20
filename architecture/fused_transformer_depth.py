from architecture.unet_blocks import DoubleConv, Down, Up, OutConv
from torch import nn
import torch
from architecture.transformer import Block, SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop, n_layer):
        super(TransformerBlock, self).__init__()
        # blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        return x


class FusionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FusionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.dc1 = DoubleConv(n_channels, 64)
        self.dc2 = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.outc = OutConv(64, n_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_classes, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.transformer1 = TransformerBlock(n_embd=64, n_head=2, block_exp=4, attn_pdrop=0.1, resid_pdrop=0.1, n_layer=8)
        
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1048),
            nn.ReLU(),
            nn.Linear(1048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        

    def forward(self, x, inverse_x, depth_x, pixel_values=None):
        # Inverse
        inv = self.conv(inverse_x)
        depth = self.conv(depth_x)
        
        x1 = self.dc1(x)
        x1_2 = self.dc2(x1)
        x2 = self.down1(x1_2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        feat1 = x5.view(x5.size(0), x5.size(1), -1)
        trans1 = self.transformer1(feat1)
        ftrans1 = self.flatten(trans1)
        
        feat2 = inv.view(x5.size(0), x5.size(1), -1)
        trans2 = self.transformer1(feat2)
        ftrans2 = self.flatten(trans2)
        
        # x6 = self.flatten(x5).view(x5.size(0), 1, -1)

        feat3 = depth.view(x5.size(0), x5.size(1), -1)
        trans3 = self.transformer1(feat3)
        ftrans3 = self.flatten(trans3)
        
        fused = torch.cat((ftrans1, ftrans2, ftrans3), dim=1)
        fused = torch.sum(fused, dim=1)
    
        seq = self.seq(fused)
    
        return seq

