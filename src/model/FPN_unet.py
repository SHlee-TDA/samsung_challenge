import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class FPN_UNet(nn.Module):
    def __init__(self):
        super(FPN_UNet, self).__init__()
        
        # Downward path (Same as original U-Net)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        
        # Upward path and lateral connections for FPN
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.lateral2 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.lateral1 = nn.Conv2d(64, 64, kernel_size=1)
        
        # FPN heads for each pyramid level
        self.fpn_out3 = nn.Conv2d(256, 13, kernel_size=3, padding=1)
        self.fpn_out2 = nn.Conv2d(128, 13, kernel_size=3, padding=1)
        self.fpn_out1 = nn.Conv2d(64, 13, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Downward path
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        
        # Upward path with lateral connections
        x = self.upconv3(x)
        conv3 = self.lateral3(conv3)
        p3 = torch.add(x, conv3)  # Element-wise addition
        out3 = self.fpn_out3(p3)
        
        x = self.upconv2(p3)
        conv2 = self.lateral2(conv2)
        p2 = torch.add(x, conv2)
        out2 = self.fpn_out2(p2)
        
        x = self.upconv1(p2)
        conv1 = self.lateral1(conv1)
        p1 = torch.add(x, conv1)
        out1 = self.fpn_out1(p1)
        
        # Note: You can return combined results or individual FPN layer outputs based on the use case.
        return out1, out2, out3

class FPN_UNet_FC(nn.Module):
    def __init__(self):
        super(FPN_UNet_FC, self).__init__()
        self.fpn_unet = FPN_UNet()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(13+13+13, 13, kernel_size=1)  # Assuming we're concatenating

    def forward(self, x):
        out1, out2, out3 = self.fpn_unet(x)

        # Upsample each output to the desired size: 224x224
        out1_upsampled = self.upsample(out1)
        out2_upsampled = self.upsample(out2)
        out3_upsampled = self.upsample(out3)

        # Concatenate the outputs along the channel dimension
        merged_output = torch.cat([out1_upsampled, out2_upsampled, out3_upsampled], dim=1)

        # Map to desired number of channels using 1x1 convolution
        final_output = self.conv1x1(merged_output)

        return final_output