import torch

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, x2, y2)] or [N, (x, y, w, h)] in absolute coords.
    Returns: [N] IoU values.
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2) if needed
    if boxes1.shape[-1] == 4 and boxes1[..., 2].max() < 2:  # Assume (x, y, w, h)
        boxes1 = torch.cat([
            boxes1[..., 0:2] - boxes1[..., 2:4] / 2,  # x1, y1
            boxes1[..., 0:2] + boxes1[..., 2:4] / 2   # x2, y2
        ], dim=-1)
        boxes2 = torch.cat([
            boxes2[..., 0:2] - boxes2[..., 2:4] / 2,
            boxes2[..., 0:2] + boxes2[..., 2:4] / 2
        ], dim=-1)

    # Intersection coordinates
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    # Intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou

# Example usage
pred_boxes = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)  # x1, y1, x2, y2
gt_boxes = torch.tensor([[60, 60, 110, 110]], dtype=torch.float32)
iou = compute_iou(pred_boxes, gt_boxes)
print("IoU:", iou.item())  # ~0.47
