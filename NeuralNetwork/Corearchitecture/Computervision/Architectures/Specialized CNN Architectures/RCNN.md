
---  

# What is R-CNN (in plain English)

**Goal:** Find *what* objects are in an image **and** *where* they are (boxes).

**Idea (2014):**

1. **Propose** a small set of *promising regions* (likely to contain objects).
2. **Crop** each region, **run a CNN** to get features,
3. **Classify** each region (cat/dog/car/background) and **refine** the box.

This was a big deal because running a CNN **only** on ~2k regions (not on every pixel) made object detection practical with CNNs.

**Pipeline (original R-CNN)**

1. **Region proposals**: Use a hand-crafted algorithm (Selective Search) to generate ~2,000 candidate boxes per image.
2. **Feature extraction**: For each proposed box, crop & warp it to a fixed size (e.g., 227×227), pass through a pretrained CNN (e.g., AlexNet) to get a feature vector.
3. **Classification**: Train one **linear SVM per class** (car vs not-car, dog vs not-dog, …).
4. **Bounding box regression**: A small linear regressor per class nudges each box to fit the object tighter.
5. **Post-processing**: Apply **Non-Maximum Suppression (NMS)** to remove duplicate boxes.

> Fast R-CNN (2015) improves step 2–4 by sharing the CNN once per image + ROI pooling.
> Faster R-CNN (2015) replaces Selective Search with a learnable **Region Proposal Network (RPN)**.
> But learning R-CNN first helps you understand them all.

---

# The core math (kept simple but complete)

## 1) Intersection over Union (IoU)

How much two boxes overlap:
[
\text{IoU}(A,B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} \in [0,1]
]

* Used to decide positives/negatives when training, and by NMS.

## 2) Box parameterization & regression

Each box is represented by center ((x, y)), width (w), height (h).
Given an **anchor/proposal** (B_a = (x_a, y_a, w_a, h_a)) and the **ground truth** (B^* = (x^*, y^*, w^*, h^*)), the regression targets are:
[
t_x^*=\frac{x^*-x_a}{w_a}, \quad
t_y^*=\frac{y^*-y_a}{h_a}, \quad
t_w^*=\log\frac{w^*}{w_a}, \quad
t_h^*=\log\frac{h^*}{h_a}
]
A regressor predicts (t=(t_x,t_y,t_w,t_h)). We convert back:
[
\hat{x} = t_x w_a + x_a,\quad
\hat{y} = t_y h_a + y_a,\quad
\hat{w} = e^{t_w} w_a,\quad
\hat{h} = e^{t_h} h_a
]

**Loss (Smooth L1 / Huber):**
[
\text{smoothL1}(z)=
\begin{cases}
0.5 z^2 & |z|<1\
|z|-0.5 & \text{otherwise}
\end{cases}
]
[
\mathcal{L}*{\text{bbox}}=\sum*{u\in{x,y,w,h}} \text{smoothL1}(t_u - t_u^*)
]

## 3) Classification (softmax + cross-entropy)

Given ROI features (\mathbf{f}), your classifier outputs logits (s_c) for classes (c\in{1..K}) plus background.
[
p(c)=\frac{e^{s_c}}{\sum_j e^{s_j}},\quad
\mathcal{L}_{\text{cls}} = -\log p(c^*)
]

## 4) Total detection loss (per ROI)

For class (c^*) and regression only when it’s a positive ROI:
[
\mathcal{L} = \mathcal{L}_{\text{cls}}

* \lambda \cdot \mathbf{1}*{[c^*\neq \text{bg}]}\cdot \mathcal{L}*{\text{bbox}}
  ]

## 5) Non-Maximum Suppression (NMS)

* Sort boxes by score (high→low).
* Keep the top box, remove any remaining boxes with IoU > threshold (e.g., 0.5) to that top box.
* Repeat.

---

# End-to-end flow (training & inference)

## Training

1. Run **Selective Search** on each training image → ~2k proposals.
2. Label proposals:

   * **Positive** if IoU with a ground-truth box ≥ 0.5 → class = that object class.
   * **Negative** if IoU ≤ 0.3 → class = background.
3. Extract CNN features for each proposal (crop/warp → CNN → feature vector).
4. Train:

   * **Classifier** (original: SVMs per class; modern toy: a softmax layer).
   * **BBox regressor** (for positives) with Smooth L1.
5. Validate; tune thresholds, NMS IoU, learning rates.

## Inference

1. Proposals (Selective Search or RPN if you’re using Faster R-CNN-style).
2. For each proposal → CNN features → class scores + bbox deltas.
3. Apply deltas to refine boxes; remove low-score ones.
4. Run **NMS** per class.
5. Keep the remaining boxes as detections.

---

# A “toy” R-CNN in PyTorch (readable and minimal)

This is **educational** code: it uses a pretrained CNN as a feature extractor, a tiny head for classification + bbox regression, and helper functions for IoU/NMS. It assumes you have proposals (e.g., from Selective Search saved to disk) and ground truth boxes.

> Tip: In practice you’ll use **Faster R-CNN** in `torchvision.models` which is battle-tested. This code is to demystify the parts.

```python
# rcnn_toy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align

# ---------- Helpers ----------
def box_iou(boxes1, boxes2):
    # boxes: [N,4] in (x1,y1,x2,y2)
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)

    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])   # [N,M,2]
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])   # [N,M,2]

    wh = (rb - lt).clamp(min=0)                       # [N,M,2]
    inter = wh[:,:,0]*wh[:,:,1]
    union = area1[:,None] + area2 - inter
    return inter / (union + 1e-6)

def nms(boxes, scores, iou_thresh=0.5):
    # Simple NMS for one class
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)

# Convert (x_c,y_c,w,h) <-> (x1,y1,x2,y2)
def xywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)

def xyxy_to_xywh(boxes):
    x1,y1,x2,y2 = boxes.unbind(-1)
    w = (x2-x1).clamp(min=1e-6)
    h = (y2-y1).clamp(min=1e-6)
    x = x1 + w/2
    y = y1 + h/2
    return torch.stack([x,y,w,h], dim=-1)

def encode_targets(anchors_xywh, gt_xywh):
    # t* = ((x*-xa)/wa, (y*-ya)/ha, log(w*/wa), log(h*/ha))
    xa,ya,wa,ha = anchors_xywh.unbind(-1)
    xg,yg,wg,hg = gt_xywh.unbind(-1)
    tx = (xg - xa) / wa
    ty = (yg - ya) / ha
    tw = torch.log(wg / wa)
    th = torch.log(hg / ha)
    return torch.stack([tx,ty,tw,th], dim=-1)

def decode_deltas(anchors_xywh, deltas):
    xa,ya,wa,ha = anchors_xywh.unbind(-1)
    tx,ty,tw,th = deltas.unbind(-1)
    x = tx*wa + xa
    y = ty*ha + ya
    w = wa * torch.exp(tw)
    h = ha * torch.exp(th)
    return torch.stack([x,y,w,h], dim=-1)

# ---------- Model ----------
class RCNNHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls = nn.Linear(1024, num_classes)      # includes background at index 0
        self.bbox = nn.Linear(1024, num_classes*4)   # class-specific box deltas

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.cls(x)
        bbox_deltas = self.bbox(x)  # [N, num_classes*4]
        return logits, bbox_deltas

class ToyRCNN(nn.Module):
    def __init__(self, num_classes=21, roi_size=7):
        super().__init__()
        # Backbone: ResNet-50 up to conv4 (C4) to keep it small
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-3])  # stride 16
        self.out_channels = 1024

        self.roi_size = roi_size
        self.head = RCNNHead(in_dim=self.out_channels*self.roi_size*self.roi_size,
                             num_classes=num_classes)

    def forward(self, images, proposals):
        """
        images: list of tensors [3,H,W] normalized to ImageNet
        proposals: list of tensors [R_i, 4] in xyxy absolute pixel coords
        """
        device = images[0].device
        # 1) Extract feature maps once per image
        feats = [self.backbone(img.unsqueeze(0)) for img in images]  # list of [1,C,H',W']
        # 2) ROI Align to fixed size
        rois = []
        roi_batch_ix = []
        for b, (fmap, boxes) in enumerate(zip(feats, proposals)):
            # format for roi_align: boxes as [idx, x1, y1, x2, y2] in input image coords scaled to fmap
            # roi_align will handle feature sampling; we just pack batch index.
            if boxes.numel() == 0:
                continue
            roi_batch_ix.append(torch.full((boxes.size(0),1), b, device=device))
            rois.append(torch.cat([roi_batch_ix[-1], boxes.to(device)], dim=1))
        rois = torch.cat(rois, dim=0) if len(rois)>0 else torch.zeros((0,5), device=device)

        # Concatenate feature maps for roi_align
        feat_batch = torch.cat(feats, dim=0)  # [B,C,H',W']

        pooled = roi_align(
            feat_batch, rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=feat_batch.shape[-1] / images[0].shape[-1],  # approx
            sampling_ratio=2, aligned=True
        )
        pooled = pooled.view(pooled.size(0), -1)  # [N, C*7*7]
        logits, bbox_deltas = self.head(pooled)   # [N,num_classes], [N, num_classes*4]
        return logits, bbox_deltas
```

### Loss (classification + bbox)

```python
def rcnn_losses(logits, bbox_deltas, labels, anchors_xyxy, gt_boxes_xyxy):
    """
    logits: [N, K] (K includes background=0)
    bbox_deltas: [N, K*4]
    labels: [N] int in [0..K-1]
    anchors_xyxy: [N,4] proposals
    gt_boxes_xyxy: [N,4] matched gt (meaningful only if label>0)
    """
    # 1) classification loss
    cls_loss = F.cross_entropy(logits, labels)

    # 2) bbox regression for positives only
    pos_mask = labels > 0
    if pos_mask.any():
        K = logits.size(1)
        anchors_xywh = xyxy_to_xywh(anchors_xyxy[pos_mask])
        gt_xywh = xyxy_to_xywh(gt_boxes_xyxy[pos_mask])
        targets = encode_targets(anchors_xywh, gt_xywh)          # [P,4]

        # pick class-specific deltas
        # bbox_deltas: [N, K*4] -> [N, K, 4]
        deltas = bbox_deltas[pos_mask].view(-1, K, 4)
        # gather by class index
        sel = labels[pos_mask].view(-1,1,1).expand(-1,1,4)
        deltas = deltas.gather(1, sel).squeeze(1)                # [P,4]

        reg_loss = F.smooth_l1_loss(deltas, targets, reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=logits.device)

    return cls_loss, reg_loss
```

### Inference (decode + per-class NMS)

```python
@torch.no_grad()
def rcnn_infer(model, image, proposals_xyxy, score_thresh=0.5, nms_thresh=0.5):
    model.eval()
    logits, bbox_deltas = model([image], [proposals_xyxy])
    probs = logits.softmax(dim=1)  # [R,K]
    K = probs.size(1)

    anchors_xywh = xyxy_to_xywh(proposals_xyxy)
    all_boxes, all_scores, all_labels = [], [], []

    for c in range(1, K):  # skip background (0)
        cls_scores = probs[:, c]
        keep = cls_scores >= score_thresh
        if not keep.any(): 
            continue

        deltas = bbox_deltas.view(-1, K, 4)[:, c, :]  # [R,4]
        boxes_c = decode_deltas(anchors_xywh, deltas)
        boxes_c = xywh_to_xyxy(boxes_c).clamp(min=0)

        boxes_c = boxes_c[keep]
        scores_c = cls_scores[keep]

        keep_idx = nms(boxes_c, scores_c, iou_thresh=nms_thresh)
        all_boxes.append(boxes_c[keep_idx])
        all_scores.append(scores_c[keep_idx])
        all_labels.append(torch.full((keep_idx.numel(),), c, dtype=torch.long))

    if len(all_boxes)==0:
        return torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
```

---

# How you’d actually train this (step-by-step)

1. **Prepare data**

   * Images + ground-truth boxes + labels (e.g., PASCAL VOC).
   * Precompute **proposals** (Selective Search). Save them per image.

2. **Build minibatches of ROIs**

   * Sample per image: e.g., 128 ROIs (25% positives, 75% negatives).
   * Match each ROI to a GT box (by highest IoU).
   * Assign **label**: foreground if IoU≥0.5 (its class), background if IoU≤0.3, ignore otherwise.
   * Store the matched GT box for positives (for bbox loss).

3. **Forward pass**

   * Run backbone once per image, ROI Align each ROI, pass through head → logits, bbox_deltas.

4. **Loss & backprop**

   * `cls_loss + λ * reg_loss` (λ ~ 1).
   * Optimizer: SGD or AdamW.

5. **Validation & tuning**

   * AP/mAP at IoU 0.5 (VOC) or [0.5:0.95] (COCO).
   * Tune score thresholds, NMS IoU.

> To keep experiments simple, use a **single class + background** first (e.g., “person”), then scale to multi-class.

---

# Common gotchas (and fixes)

* **Box coordinate mistakes**: Keep conversions `xyxy ↔ xywh` consistent.
* **Class imbalance**: Background massively outnumbers positives → use ROI sampling with a pos:neg ratio (e.g., 1:3).
* **Inaccurate spatial scale in roi_align**: Prefer ROI Align on feature maps with known stride (e.g., 16).
* **Exploding deltas**: Clip predicted deltas before decoding (e.g., `tw, th ∈ [-5,5]`).
* **Low recall** from Selective Search: You can lower the positive IoU during training to 0.5 and keep more proposals; later switch to Faster R-CNN for better proposals.

---

# Where R-CNN fits vs. Fast/Faster R-CNN

* **R-CNN**: Proposals (hand-crafted) → crop+CNN per ROI → SVMs + bbox regressor. Slow (per-ROI CNN), multi-stage training.
* **Fast R-CNN**: Run CNN **once** per image → **ROI Pooling** → single network head (softmax + bbox). Much faster, end-to-end (except proposals).
* **Faster R-CNN**: Adds **RPN** (small CNN) to generate proposals → fully learnable, fast, strong baseline still today.

If you just want results: use `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)` and finetune. If you want to *learn the internals*, the toy R-CNN above is perfect.

---

# Minimal “from-scratch” checklist to practice

1. Load a tiny dataset (10–50 images) with one class (say, “person”).
2. Precompute proposals (or use a few hand-made boxes to start).
3. Train the toy head for 5–10 epochs.
4. Visualize predictions + NMS.
5. Add a second class; watch classification and regression losses separately.
6. Swap proposals for an RPN (you’re now at Faster R-CNN!).

---

