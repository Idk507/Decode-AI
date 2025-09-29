import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple C3 module (like YOLOv11's C3k2)
class C3Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C3Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([x, self.conv1(residual)], dim=1)
        x = self.conv3(x)
        x = self.bn(x)
        return self.act(x)

# Simple Backbone
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)  # 320x320
        self.c3_1 = C3Module(64, 128)  # 320x320
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 160x160
        self.c3_2 = C3Module(128, 256)  # 160x160
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 80x80
        self.c3_3 = C3Module(256, 512)  # 80x80

    def forward(self, x):
        x1 = self.c3_1(self.conv1(x))
        x2 = self.c3_2(self.conv2(x1))
        x3 = self.c3_3(self.conv3(x2))
        return x1, x2, x3  # Multi-scale features

# Simple Neck (FPN-like)
class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.c3_1 = C3Module(512, 256)  # Fuse 160x160 + 80x80
        self.conv2 = nn.Conv2d(256, 128, 1)
        self.c3_2 = C3Module(256, 128)  # Fuse 320x320 + 160x160

    def forward(self, x1, x2, x3):
        x3 = self.conv1(x3)  # 80x80
        x3_up = self.up(x3)  # 160x160
        x2 = self.c3_1(torch.cat([x2, x3_up], dim=1))  # Fuse
        x2_up = self.up(self.conv2(x2))  # 320x320
        x1 = self.c3_2(torch.cat([x1, x2_up], dim=1))  # Fuse
        return x1, x2, x3

# Detection Head (Anchor-Free)
class Head(nn.Module):
    def __init__(self, num_classes=80, strides=[8, 16, 32]):
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleList([
            nn.Conv2d(c, 4 + 1 + num_classes, 1) for c in [128, 256, 512]
        ])
        self.strides = strides

    def forward(self, features):
        outputs = []
        for i, (feat, head) in enumerate(zip(features, self.heads)):
            pred = head(feat)  # Shape: [B, 4+1+C, H, W]
            outputs.append(pred)
        return outputs

# Post-Processing
def post_process(preds, conf_thres=0.5, iou_thres=0.5, num_classes=80):
    boxes, scores, classes = [], [], []
    for pred, stride in zip(preds, [8, 16, 32]):
        B, _, H, W = pred.shape
        pred = pred.sigmoid()  # Apply sigmoid to objectness/class scores
        pred = pred.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # [B, H*W, 4+1+C]

        # Extract components
        box = pred[..., :4]  # x, y, w, h
        obj = pred[..., 4:5]  # Objectness
        cls = pred[..., 5:]  # Class probs

        # Convert to absolute coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x.to(pred.device).view(1, H*W, 1)
        grid_y = grid_y.to(pred.device).view(1, H*W, 1)
        box[..., 0:2] = (box[..., 0:2] + torch.cat([grid_x, grid_y], dim=-1)) * stride  # Center
        box[..., 2:4] = torch.exp(box[..., 2:4]) * stride  # Width, height

        # Filter by confidence
        conf = obj * cls.max(dim=-1, keepdim=True)[0]
        mask = conf > conf_thres
        box = box[mask.expand_as(box)].reshape(-1, 4)
        conf = conf[mask].reshape(-1)
        cls = cls[mask.expand_as(cls)].argmax(dim=-1).reshape(-1)

        boxes.append(box)
        scores.append(conf)
        classes.append(cls)

    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)
    classes = torch.cat(classes, dim=0)

    # Apply NMS
    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    return boxes[keep], scores[keep], classes[keep]

# Full YOLO Model
class YOLO(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLO, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes=num_classes)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        x1, x2, x3 = self.neck(x1, x2, x3)
        preds = self.head([x1, x2, x3])
        return preds

# Example Usage
if __name__ == "__main__":
    import torchvision

    # Initialize model
    model = YOLO(num_classes=80).eval()
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Dummy input (batch_size=1, channels=3, height=640, width=640)
    input_image = torch.randn(1, 3, 640, 640).to(model.device)

    # Forward pass
    with torch.no_grad():
        preds = model(input_image)

    # Post-process predictions
    boxes, scores, classes = post_process(preds, conf_thres=0.5, iou_thres=0.5)

    print(f"Detected {len(boxes)} objects")
    for box, score, cls in zip(boxes, scores, classes):
        print(f"Box: {box.tolist()}, Score: {score.item():.2f}, Class: {cls.item()}")
