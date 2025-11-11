# segnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(SegNet, self).__init__()

        # Encoder (like VGG16 but smaller)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)

        # Decoder (mirror)
        self.dec3 = self._block(256, 128)
        self.dec2 = self._block(128, 64)
        self.dec1 = self._block(64, num_classes, last=True)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def _block(self, in_ch, out_ch, last=False):
        if not last:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            # Last block: single conv to get class logits
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.enc1(x)
        x1p, idx1 = self.pool(x1)

        x2 = self.enc2(x1p)
        x2p, idx2 = self.pool(x2)

        x3 = self.enc3(x2p)
        x3p, idx3 = self.pool(x3)

        # ---- Decoder ----
        x3u = self.unpool(x3p, idx3, output_size=x3.size())
        x3d = self.dec3(x3u)

        x2u = self.unpool(x3d, idx2, output_size=x2.size())
        x2d = self.dec2(x2u)

        x1u = self.unpool(x2d, idx1, output_size=x1.size())
        out = self.dec1(x1u)  # logits [B, num_classes, H, W]

        return out

model = SegNet(in_channels=3, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()  # expects targets [B,H,W] with class indices
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for img, mask in train_loader:
    img, mask = img.to(device), mask.to(device)
    logits = model(img)
    loss = criterion(logits, mask.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    logits = model(img.unsqueeze(0).to(device))
    preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

