import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant

class RepConv(nn.Module):  # Reparameterized Conv for edge
    def __init__(self, in_channels, out_channels):
        super(RepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.silu(self.bn(self.conv1(x) + self.conv3(x)))  # Multi-branch train, single-path infer

    def rep_param(self):  # Reparameterize for edge inference
        kernel3 = self.conv3.weight.sum(dim=(2,3), keepdim=True)
        self.conv1.weight.data += kernel3 * self.conv3.weight.new_ones(self.conv1.weight.shape[0], self.conv1.weight.shape[1], 1, 1)
        self.conv3 = nn.Identity()  # Simplify at inference

class EfficientRepBlock(nn.Module):  # YOLOv6 Backbone Block
    def __init__(self, channels):
        super(EfficientRepBlock, self).__init__()
        self.rep1 = RepConv(channels, channels)
        self.rep2 = RepConv(channels, channels)

    def forward(self, x):
        return self.rep2(self.rep1(x))

class EELANBlock(nn.Module):  # YOLOv7 Neck-like Block
    def __init__(self, channels):
        super(EELANBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, 1)
        self.conv2 = nn.Conv2d(channels//2, channels, 3, padding=1)
        self.rep_head = RepConv(channels, channels)  # Reparameterized Head-like

    def forward(self, x):
        return self.rep_head(x + self.conv2(self.conv1(x)))  # Residual + Rep

class YOLOv6v7Head(nn.Module):  # Reparameterized Decoupled Head (YOLOv6/YOLOv7)
    def __init__(self, num_classes=80, anchors_per_scale=3):
        super(YOLOv6v7Head, self).__init__()
        self.cls_conv = RepConv(256, num_classes * anchors_per_scale)  # Class branch
        self.reg_conv = RepConv(256, 5 * anchors_per_scale)  # Reg branch (x,y,w,h,obj)

    def forward(self, x):
        cls = torch.sigmoid(self.cls_conv(x))
        reg = torch.sigmoid(self.reg_conv(x))  # Simplified for edge
        return reg, cls

class YOLOv6v7Model(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv6v7Model, self).__init__()
        self.backbone = nn.Sequential(  # EfficientRep-like
            RepConv(3, 32), EfficientRepBlock(32), RepConv(32, 64),
            EfficientRepBlock(64), RepConv(64, 128)  # Simplified scales
        )
        self.neck = nn.Sequential(  # E-ELAN-like
            EELANBlock(128), RepConv(128, 256)
        )
        self.head = YOLOv6v7Head(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fused = self.neck(features)
        return self.head(fused)

    def quantize_for_edge(self):  # Simulate edge optimization
        self = quant.quantize_dynamic(self, {nn.Conv2d: torch.nn.quantized.dynamic.modules.Conv2d}, dtype=torch.qint8)
        for m in self.modules():
            if hasattr(m, 'rep_param'):
                m.rep_param()  # Reparameterize
        return self

# Example usage
model = YOLOv6v7Model(num_classes=80)
input_img = torch.randn(1, 3, 640, 640)
output = model(input_img)
print("Output shapes:", [o.shape for o in output])
model.quantize_for_edge()  # Edge prep
print("Model quantized for edge.")
