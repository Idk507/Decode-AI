import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3Anchor(nn.Module):
    def __init__(self, S=13, B=3, C=80, anchors=None):
        super(YOLOv3Anchor, self).__init__()
        self.S = S  # Grid size (e.g., 13x13)
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes
        self.anchors = torch.tensor(anchors if anchors else [[10,13], [16,30], [33,23]], dtype=torch.float32)  # Example anchors
        
        # Simple backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # Head: Output SxSx(B*5 + C)
        self.head = nn.Conv2d(256, B * (5 + C), kernel_size=1)

    def forward(self, x):
        # Backbone
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        # Head
        x = self.head(x)  # Shape: (batch, B*(5+C), S, S)
        x = x.permute(0, 2, 3, 1)  # Reshape to (batch, S, S, B*(5+C))
        # Split predictions
        preds = x.view(-1, self.S, self.S, self.B, 5 + self.C)
        boxes = preds[..., :4]  # tx, ty, tw, th
        conf = preds[..., 4:5]  # Objectness
        cls = preds[..., 5:]    # Class probs
        # Transform box predictions
        grid_x = torch.linspace(0, self.S-1, self.S).view(1, 1, self.S, 1).expand(-1, self.S, -1, self.B)
        grid_y = grid_x.permute(0, 2, 1, 3)
        anchors = self.anchors.to(x.device).view(1, 1, 1, self.B, 2)
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2]) + torch.cat((grid_x, grid_y), dim=-1)  # bx, by
        boxes[..., 2:4] = torch.exp(boxes[..., 2:4]) * anchors  # bw, bh
        conf = torch.sigmoid(conf)
        cls = torch.sigmoid(cls)
        return boxes, conf, cls

# Simplified loss (CIoU not implemented for brevity)
def yolo_anchor_loss(preds, targets, lambda_coord=5.0, lambda_noobj=0.5):
    boxes_pred, conf_pred, cls_pred = preds
    boxes_gt, conf_gt, cls_gt = targets
    obj_mask = conf_gt > 0
    # Coordinate loss
    coord_loss = lambda_coord * torch.sum((boxes_pred[obj_mask] - boxes_gt[obj_mask]) ** 2)
    # Confidence loss
    conf_loss_obj = torch.sum((conf_pred[obj_mask] - conf_gt[obj_mask]) ** 2)
    conf_loss_noobj = lambda_noobj * torch.sum((conf_pred[~obj_mask] - conf_gt[~obj_mask]) ** 2)
    # Class loss
    cls_loss = torch.sum((cls_pred[obj_mask] - cls_gt[obj_mask]) ** 2)
    return coord_loss + conf_loss_obj + conf_loss_noobj + cls_loss

# Example usage
model = YOLOv3Anchor(S=13, B=3, C=80)
input_img = torch.randn(1, 3, 416, 416)
boxes, conf, cls = model(input_img)
# Dummy targets
targets = (
    torch.randn(1, 13, 13, 3, 4),  # boxes
    torch.randn(1, 13, 13, 3, 1),  # confidences
    torch.randn(1, 13, 13, 80)     # classes
)
loss = yolo_anchor_loss((boxes, conf, cls), targets)
print("Loss:", loss.item())anchor
