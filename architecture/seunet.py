from architecture.unet_blocks import DoubleConv, Down, Up, OutConv
from torch import nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
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
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dc4 = DoubleConv(64, 64)
        self.outc = OutConv(64, n_classes)
        
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
        

    def forward(self, x):
        x1 = self.dc1(x)
        x1_2 = self.dc2(x1)
        # print(x1.shape)
        # print(x1_2.shape)
        x2 = self.down1(x1_2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.flatten(x5)
        seq = self.seq(x6)
        # print(seq.shape)
        
        ## Downsized U-Net
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1_2)
        x = self.dc4(x)
        logits = self.outc(x)
        
        return seq