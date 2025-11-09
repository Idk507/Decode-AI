import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
# Assume you have selective_search library or mock proposals

class SimpleRCNN(nn.Module):
    def __init__(self, num_classes=21):  # e.g., 20 VOC + bg
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)  # Feature extractor
        self.cnn.fc = nn.Identity()  # Remove classifier, get 2048-dim features
        self.classifier = nn.Linear(2048, num_classes)  # Softmax for classes
        self.bbox_reg = nn.Linear(2048, 4 * num_classes)  # 4 deltas per class
    
    def forward(self, patches):  # patches: list of tensor images
        features = []
        for patch in patches:
            feat = self.cnn(patch)  # Shape: [1, 2048]
            features.append(feat)
        feats = torch.stack(features)  # [N, 2048]
        
        class_scores = self.classifier(feats)  # [N, num_classes]
        bbox_deltas = self.bbox_reg(feats).view(-1, num_classes, 4)  # [N, num_classes, 4]
        
        return class_scores, bbox_deltas

# Mock function for proposals (in real: use selective_search)
def get_proposals(image):
    # Dummy: return 5 random boxes as (x,y,w,h) normalized
    h, w = image.size[1], image.size[0]
    return [(np.random.rand(4) * [w, h, w*0.5, h*0.5]).astype(int) for _ in range(5)]

# Warp patch (simple crop+resize)
def warp_patch(image, box):
    x, y, w, h = box
    patch = image.crop((x, y, x+w, y+h))
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485], [0.229])])
    return transform(patch).unsqueeze(0)

# Example usage
model = SimpleRCNN()
model.eval()

# Load sample image (replace with your path)
img = Image.open("sample.jpg")  # Assume RGB image
proposals = get_proposals(img)

patches = [warp_patch(img, prop) for prop in proposals]
with torch.no_grad():
    scores, deltas = model(patches)

# Softmax for probs
probs = torch.softmax(scores, dim=1)
print("Class probs (last proposal):", probs[0])  # e.g., [bg, cat, dog, ...]

# For NMS: Use torchvision.ops.nms (filter high-score boxes, suppress overlaps)
# Refined box example: for class 1 (cat), box + deltas[0,1,:]
