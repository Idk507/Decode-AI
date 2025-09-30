import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleYOLOGrid(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(SimpleYOLOGrid, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Boxes per cell
        self.C = C  # Number of classes
        
        # Simple backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # Head: Output SxSx(B*5 + C)
        self.conv3 = nn.Conv2d(192, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512 * (S // 4) * (S // 4), S * S * (B * 5 + C))

    def forward(self, x):
        # Backbone
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x), 0.1)
        # Flatten and head
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        # Split predictions
        boxes = x[..., :self.B * 4].view(-1, self.S, self.S, self.B, 4)  # x, y, w, h
        confidences = x[..., self.B * 4:self.B * 5].view(-1, self.S, self.S, self.B)
        classes = x[..., self.B * 5:].view(-1, self.S, self.S, self.C)
        # Apply sigmoid to normalize
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # x, y to [0,1]
        confidences = torch.sigmoid(confidences)  # Objectness to [0,1]
        classes = torch.sigmoid(classes)  # Class probs
        return boxes, confidences, classes

# Loss function (simplified)
def yolo_grid_loss(preds, targets, lambda_coord=5.0, lambda_noobj=0.5):
    boxes_pred, conf_pred, cls_pred = preds
    boxes_gt, conf_gt, cls_gt = targets  # Same shapes
    # Coordinate loss (only for cells with objects)
    obj_mask = conf_gt > 0
    coord_loss = lambda_coord * torch.sum((boxes_pred[obj_mask] - boxes_gt[obj_mask]) ** 2)
    # Confidence loss
    conf_loss_obj = torch.sum((conf_pred[obj_mask] - conf_gt[obj_mask]) ** 2)
    conf_loss_noobj = lambda_noobj * torch.sum((conf_pred[~obj_mask] - conf_gt[~obj_mask]) ** 2)
    # Class loss
    cls_loss = torch.sum((cls_pred[obj_mask] - cls_gt[obj_mask]) ** 2)
    return coord_loss + conf_loss_obj + conf_loss_noobj + cls_loss

# Example usage
model = SimpleYOLOGrid(S=7, B=2, C=20)
input_img = torch.randn(1, 3, 448, 448)
boxes, conf, cls = model(input_img)
# Dummy targets for loss
targets = (
    torch.randn(1, 7, 7, 2, 4),  # boxes
    torch.randn(1, 7, 7, 2),     # confidences
    torch.randn(1, 7, 7, 20)     # classes
)
loss = yolo_grid_loss((boxes, conf, cls), targets)
print("Loss:", loss.item())
