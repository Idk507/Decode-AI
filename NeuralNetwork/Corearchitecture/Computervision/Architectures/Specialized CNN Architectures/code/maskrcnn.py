# pip install torch torchvision
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = maskrcnn_resnet50_fpn(pretrained=True)   # pretrained on COCO (detect + mask)
model.to(device)
model.eval()   # or .train() for fine-tuning

# Inference example (one image)
from PIL import Image
img = Image.open("example.jpg").convert("RGB")
img_tensor = F.to_tensor(img).to(device)  # [3,H,W], values 0..1
with torch.no_grad():
    outputs = model([img_tensor])  # list of dicts
out = outputs[0]
# out contains keys: boxes, labels, scores, masks (N,1,H,W)
boxes = out['boxes']
labels = out['labels']
scores = out['scores']
masks = out['masks']  # float masks in [0,1] (per-pixel)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMaskHead(nn.Module):
    def __init__(self, in_channels=256, conv_channels=256, num_classes=21, mask_size=28):
        super().__init__()
        # four 3x3 conv layers (as in Mask R-CNN)
        self.conv1 = nn.Conv2d(in_channels, conv_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        # deconv to upsample by 2
        self.deconv = nn.ConvTranspose2d(conv_channels, conv_channels, 2, stride=2)
        # final 1x1 conv to produce K masks
        self.mask_pred = nn.Conv2d(conv_channels, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.mask_size = mask_size

    def forward(self, x):
        # x: [N, C, H, W] e.g., H,W might be 7x7 from RoIAlign
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.deconv(x))    # upsample (7x7 -> 14x14)
        x = self.mask_pred(x)            # [N, K, 14, 14]
        # If we want 28x28, another upsample or set deconv stride/size accordingly
        x = F.interpolate(x, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        # per-pixel logits -> use sigmoid in loss
        return x  # raw logits, shape [N, K, mask_size, mask_size]
