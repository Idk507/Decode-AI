import torch

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, x2, y2)] or [N, (x, y, w, h)].
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

    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

def nms(boxes, scores, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression.
    boxes: [N, (x1, y1, x2, y2)] or [N, (x, y, w, h)].
    scores: [N] confidence scores.
    iou_threshold: IoU threshold for suppression.
    Returns: Indices of kept boxes.
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by scores
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        # Keep highest-scoring box
        keep.append(order[0].item())
        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[0:1], boxes[1:])
        # Keep boxes with IoU < threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]

    return torch.tensor(keep, dtype=torch.long)

# Example usage
boxes = torch.tensor([
    [50, 50, 100, 100],  # Box 1
    [60, 60, 110, 110],  # Box 2 (overlaps with Box 1)
    [200, 200, 250, 250]  # Box 3 (no overlap)
], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7])  # Confidence scores
keep_indices = nms(boxes, scores, iou_threshold=0.45)
print("Kept indices:", keep_indices)  # Should keep Box 1 and Box 3
