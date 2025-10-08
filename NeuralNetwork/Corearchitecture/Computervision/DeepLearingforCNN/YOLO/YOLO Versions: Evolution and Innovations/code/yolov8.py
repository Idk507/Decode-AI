import torch
import torch.nn as nn
import torch.nn.functional as F

class C2fBlock(nn.Module):
    def __init__(self, c_in, c_out, shortcut=True):
        super(C2fBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out // 2, 1)
        self.bn1 = nn.BatchNorm2d(c_out // 2)
        self.conv2 = nn.Conv2d(c_out // 2, c_out // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out // 2)
        self.conv3 = nn.Conv2d(c_in, c_out // 2, 1)
        self.bn3 = nn.BatchNorm2d(c_out // 2)
        self.shortcut = shortcut

    def forward(self, x):
        x1 = F.silu(self.bn1(self.conv1(x)))
        x2 = F.silu(self.bn2(self.conv2(x1)))
        x3 = F.silu(self.bn3(self.conv3(x)))
        x = torch.cat([x2, x3], dim=1)
        if self.shortcut:
            x += x3
        return x

class C2fDarknet(nn.Module):
    def __init__(self):
        super(C2fDarknet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 640x640 -> 320x320
        self.bn2 = nn.BatchNorm2d(64)
        self.c2f1 = C2fBlock(64, 128)  # 320x320
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 320x320 -> 160x160
        self.bn3 = nn.BatchNorm2d(256)
        self.c2f2 = C2fBlock(256, 512)  # 160x160 -> 80x80
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # 80x80 -> 40x40
        self.bn4 = nn.BatchNorm2d(512)
        self.c2f3 = C2fBlock(512, 1024)  # 40x40 -> 20x20
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)  # 20x20
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        C3 = self.c2f1(x)  # 320x320
        x = F.silu(self.bn3(self.conv3(C3)))
        C4 = self.c2f2(x)  # 80x80
        x = F.silu(self.bn4(self.conv4(C4)))
        C5 = self.c2f3(x)  # 40x40
        C5 = F.silu(self.bn5(self.conv5(C5)))  # 20x20
        return [C3, C4, C5]

class SPPF(nn.Module):
    def __init__(self, in_channels):
        super(SPPF, self).__init__()
        self.pool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = x
        x2 = self.pool1(x)
        x3 = self.pool2(x2)
        x4 = self.pool3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return F.silu(self.conv(x))

class PANet(nn.Module):
    def __init__(self, in_channels=[128, 512, 1024]):
        super(PANet, self).__init__()
        self.sppf = SPPF(in_channels[2])
        self.lateral1 = nn.Conv2d(in_channels[2], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[0], 256, kernel_size=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        C3, C4, C5 = features
        P5 = self.sppf(C5)
        P5 = F.silu(self.lateral1(P5))  # 20x20
        P4 = F.silu(self.lateral2(C4)) + F.interpolate(P5, scale_factor=2, mode='nearest')  # 40x40
        P3 = F.silu(self.lateral3(C3)) + F.interpolate(P4, scale_factor=2, mode='nearest')  # 80x80
        P4 = F.silu(self.smooth2(P4)) + F.silu(self.down1(P3))  # 40x40
        P5 = F.silu(self.smooth1(P5)) + F.silu(self.down2(P4))  # 20x20
        P3 = F.silu(self.smooth3(P3))  # 80x80
        return [P3, P4, P5]

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes=80, for_segmentation=False):
        super(YOLOv8Head, self).__init__()
        self.num_classes = num_classes
        self.for_segmentation = for_segmentation
        self.cls_conv = nn.Conv2d(256, num_classes, kernel_size=1)  # Classification branch
        self.reg_conv = nn.Conv2d(256, 4, kernel_size=1)  # Regression branch (anchor-free)
        self.obj_conv = nn.Conv2d(256, 1, kernel_size=1)  # Objectness branch
        self.mask_conv = nn.Conv2d(256, 32, kernel_size=1) if for_segmentation else nn.Identity()  # Segmentation branch

    def forward(self, x):
        cls = torch.sigmoid(self.cls_conv(x))  # Class probabilities
        reg = self.reg_conv(x)  # Direct box predictions (x, y, w, h)
        obj = torch.sigmoid(self.obj_conv(x))  # Objectness
        outputs = [reg, obj, cls]
        if self.for_segmentation:
            mask = self.mask_conv(x)  # Mask coefficients
            outputs.append(mask)
        return outputs

class YOLOv8(nn.Module):
    def __init__(self, S=[20, 40, 80], num_classes=80, for_segmentation=False):
        super(YOLOv8, self).__init__()
        self.S = S
        self.backbone = C2fDarknet()
        self.neck = PANet(in_channels=[128, 512, 1024])
        self.heads = nn.ModuleList([
            YOLOv8Head(num_classes, for_segmentation) for _ in range(3)
        ])

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(features)
        outputs = [self.heads[i](fused_features[i]) for i in range(3)]
        return outputs

# Simplified Mosaic augmentation
def mosaic_augmentation(images, size=640):
    batch_size, _, h, w = images.shape
    mosaic = torch.zeros(batch_size, 3, size, size)
    for i in range(batch_size):
        img1, img2, img3, img4 = [F.interpolate(images[i:i+1], size=(size//2, size//2)) for _ in range(4)]
        mosaic[i, :, 0:size//2, 0:size//2] = img1
        mosaic[i, :, 0:size//2, size//2:size] = img2
        mosaic[i, :, size//2:size, 0:size//2] = img3
        mosaic[i, :, size//2:size, size//2:size] = img4
    return mosaic

# Simplified loss (CIoU + DFL for detection, BCE + Dice for segmentation)
def yolov8_loss(preds, targets):
    loss = 0
    for (reg_pred, obj_pred, cls_pred, *mask_pred), (boxes_gt, obj_gt, cls_gt, *mask_gt) in zip(preds, targets):
        obj_mask = obj_gt[..., 0] > 0
        # Box loss (CIoU placeholder with squared error)
        box_loss = 5.0 * torch.sum((reg_pred[obj_mask] - boxes_gt[obj_mask]) ** 2)
        # Objectness and classification loss
        obj_loss = F.binary_cross_entropy(obj_pred, obj_gt, reduction='sum')
        cls_loss = F.binary_cross_entropy(cls_pred[obj_mask], cls_gt[obj_mask], reduction='sum')
        # Segmentation loss (if applicable)
        seg_loss = 0
        if mask_pred:
            seg_loss = F.binary_cross_entropy(mask_pred[obj_mask], mask_gt[obj_mask], reduction='sum')
        loss += box_loss + obj_loss + cls_loss + seg_loss
    return loss

# Example usage
model = YOLOv8(S=[20, 40, 80], num_classes=80, for_segmentation=True)
input_img = torch.randn(4, 3, 640, 640)
input_img = mosaic_augmentation(input_img)  # Apply Mosaic
outputs = model(input_img)
for i, (reg, obj, cls, *mask) in enumerate(outputs):
    print(f"Scale {i+1}: Box shape: {reg.shape}, Objectness shape: {obj.shape}, Classes shape: {cls.shape}, Mask shape: {mask[0].shape if mask else None}")
# Dummy targets
targets = [
    (torch.randn(4, S, S, 4), torch.randn(4, S, S, 1), torch.randn(4, S, S, 80), torch.randn(4, S, S, 32))
    for S in [80, 40, 20]
]
loss = yolov8_loss(outputs, targets)
print("Loss:", loss.item())
