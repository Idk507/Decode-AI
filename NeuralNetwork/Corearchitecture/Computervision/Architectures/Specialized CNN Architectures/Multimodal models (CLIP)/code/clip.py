"""
clip_from_scratch.py
A compact, runnable CLIP-style implementation in PyTorch.

Features:
- Image encoder: small ConvNet (easy to replace with ResNet/ViT)
- Text encoder: token embedding + Transformer encoder (using nn.TransformerEncoder)
- Projection heads and L2 normalization
- Symmetric contrastive loss (InfoNCE) with learnable temperature
- Tiny Dataset utilities (build vocab, tokenize, loader)
- Training loop example and zero-shot inference example

Author: educational example
"""
import math
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Utilities: tokenizer + dataset
# -------------------------
class SimpleVocab:
    """Builds a small whitespace-based vocabulary from training captions."""
    def __init__(self, min_freq=1, unk_token="<unk>", pad_token="<pad>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.token2idx = {pad_token: 0, unk_token: 1}
        self.idx2token = [pad_token, unk_token]
        self.freqs = {}

    def add_text(self, text: str):
        for t in text.strip().split():
            t = t.lower()
            self.freqs[t] = self.freqs.get(t, 0) + 1

    def build(self):
        for token, f in sorted(self.freqs.items(), key=lambda x: -x[1]):
            if f < self.min_freq:
                continue
            if token in self.token2idx: 
                continue
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)

    def tokenize(self, text: str, max_len=32) -> List[int]:
        toks = []
        for t in text.strip().split():
            t = t.lower()
            idx = self.token2idx.get(t, self.token2idx[self.unk_token])
            toks.append(idx)
            if len(toks) >= max_len:
                break
        # pad
        if len(toks) < max_len:
            toks += [self.token2idx[self.pad_token]] * (max_len - len(toks))
        return toks

    def __len__(self):
        return len(self.idx2token)


class ImageTextDataset(Dataset):
    """Tiny synthetic dataset wrapper: images are tensors, captions are strings or pre-tokenized."""
    def __init__(self, images: List[torch.Tensor], captions: List[str], vocab: SimpleVocab, max_len=32, transforms=None):
        assert len(images) == len(captions)
        self.images = images
        self.captions = captions
        self.vocab = vocab
        self.max_len = max_len
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transforms:
            img = self.transforms(img)
        tokens = torch.tensor(self.vocab.tokenize(self.captions[idx], self.max_len), dtype=torch.long)
        return img, tokens

# -------------------------
# Image encoder (small ConvNet — replaceable)
# -------------------------
class SmallConvEncoder(nn.Module):
    """Very small CNN that outputs a feature vector."""
    def __init__(self, in_ch=3, hidden=64, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden*2, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(hidden*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden*2, hidden*4, kernel_size=3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(hidden*4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(hidden*4, out_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        f = self.net(x).view(x.size(0), -1)
        return self.fc(f)  # (B, out_dim)


# -------------------------
# Text encoder (embedding + Transformer encoder)
# -------------------------
class SimpleTextEncoder(nn.Module):
    """
    Text encoder: token embedding + positional embeddings + TransformerEncoder
    Returns a pooled vector per input (take embedding of <pad> masked last token or mean-pool non-pad tokens).
    """
    def __init__(self, vocab_size: int, token_dim: int = 256, max_len: int = 32, nhead: int = 8, nhid: int = 512, nlayers: int = 4, out_dim: int = 512):
        super().__init__()
        self.token_dim = token_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, token_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dim_feedforward=nhid, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pooler = nn.Linear(token_dim, out_dim)

    def forward(self, token_ids: torch.LongTensor):
        # token_ids: (B, L)
        mask = (token_ids == 0)  # padding mask for transformer (True == pad)
        x = self.embedding(token_ids) + self.pos_embed[:, :token_ids.size(1), :]
        # PyTorch transformer expects (L, B, D)
        x = x.transpose(0, 1)  # (L, B, D)
        # src_key_padding_mask expects shape (B, L) with True where padding
        x = self.transformer(x, src_key_padding_mask=mask)  # (L, B, D)
        x = x.transpose(0, 1)  # (B, L, D)
        # Simple pooling: mean over non-pad tokens
        lengths = (~mask).sum(dim=1).clamp(min=1).unsqueeze(1)  # (B,1)
        summed = (x * (~mask).unsqueeze(2).to(x.dtype)).sum(dim=1)  # (B, D)
        mean_pooled = summed / lengths  # (B, D)
        return self.pooler(mean_pooled)  # (B, out_dim)


# -------------------------
# CLIP model wrapper
# -------------------------
class CLIPModel(nn.Module):
    def __init__(self,
                 image_encoder: nn.Module,
                 text_encoder: nn.Module,
                 proj_dim: int = 256):
        """
        image_encoder: maps (B,3,H,W) -> (B, D_img)
        text_encoder: maps (B, L) -> (B, D_text)
        proj_dim: final joint embedding dimension
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        D_img = getattr(image_encoder, 'fc').out_features if hasattr(image_encoder, 'fc') else 512
        D_text = getattr(text_encoder, 'pooler').out_features if hasattr(text_encoder, 'pooler') else 512

        # projection heads
        self.img_proj = nn.Linear(D_img, proj_dim)
        self.txt_proj = nn.Linear(D_text, proj_dim)

        # logit scaling (learned temperature)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))  # typical starting tau ~0.07

    def forward(self, images: torch.Tensor, token_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (image_embeddings, text_embeddings) each L2-normalized (B, proj_dim)
        """
        img_f = self.image_encoder(images)   # (B, D_img)
        txt_f = self.text_encoder(token_ids) # (B, D_text)

        z_img = self.img_proj(img_f)        # (B, proj_dim)
        z_txt = self.txt_proj(txt_f)        # (B, proj_dim)

        # L2 normalize
        z_img = F.normalize(z_img, p=2, dim=1)
        z_txt = F.normalize(z_txt, p=2, dim=1)
        return z_img, z_txt

    def get_logits(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity logits (scaled by 1/tau). Returns (B, B) matrix where logits[i,j]
        is sim(image_i, text_j).
        """
        # z_img: (B, D), z_txt: (B, D)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (z_img @ z_txt.t())  # (B, B)
        return logits


# -------------------------
# Contrastive symmetric loss (InfoNCE both directions)
# -------------------------
def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, B) where logits[i,j] is sim(img_i, txt_j) * (1/tau)
    Return scalar symmetric loss averaged across both directions.
    """
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device, dtype=torch.long)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


# -------------------------
# Small training loop example (toy data)
# -------------------------
def train_one_epoch(model: CLIPModel, dataloader: DataLoader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, tokens in dataloader:
        images = images.to(device)
        tokens = tokens.to(device)
        z_img, z_txt = model(images, tokens)
        logits = model.get_logits(z_img, z_txt)
        loss = clip_contrastive_loss(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


# -------------------------
# Zero-shot classification helper
# -------------------------
@torch.no_grad()
def zero_shot_classify(model: CLIPModel, image: torch.Tensor, class_prompts: List[str], vocab: SimpleVocab, device, max_len=32):
    """
    Given a list of class prompt strings, returns softmax probabilities over classes for the image.
    (Builds text embeddings by tokenizing the prompts with `vocab`.)
    """
    model.eval()
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)
    # tokenize prompts
    toks = torch.stack([torch.tensor(vocab.tokenize(p, max_len), dtype=torch.long) for p in class_prompts], dim=0).to(device)
    z_img, _ = model(image, torch.zeros((1, max_len), dtype=torch.long, device=device))  # we only need image embedding; dummy tokens not used
    # compute text embeddings: pass prompts through text encoder directly (proj & normalize)
    with torch.no_grad():
        txt_f = model.text_encoder(toks)  # (C, D_text)
        z_txt = model.txt_proj(txt_f)
        z_txt = F.normalize(z_txt, p=2, dim=1)  # (C, D)
        z_img = z_img.squeeze(0)  # (D,)
        # compute logits: (C,)
        logits = (model.logit_scale.exp() * (z_txt @ z_img)).squeeze()
        probs = logits.softmax(dim=0)
    return probs.cpu().numpy(), logits.cpu().numpy()


# -------------------------
# Demo: build small synthetic dataset and run one epoch
# -------------------------
def demo_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create synthetic data: random images and toy captions
    N = 200
    H = W = 64
    images = [torch.randn(3, H, W) for _ in range(N)]
    # toy captions — keep repeated tokens so vocab isn't huge
    captions = [random.choice(["a dog", "a cat", "a car", "a person", "a tree", "a bird"]) + " on a background" for _ in range(N)]

    # Build vocab
    vocab = SimpleVocab(min_freq=1)
    for c in captions:
        vocab.add_text(c)
    vocab.build()
    print("Vocab size:", len(vocab))

    dataset = ImageTextDataset(images, captions, vocab, max_len=16)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # instantiate encoders and model
    img_enc = SmallConvEncoder(in_ch=3, hidden=32, out_dim=512)
    txt_enc = SimpleTextEncoder(vocab_size=len(vocab), token_dim=128, max_len=16, nhead=4, nhid=256, nlayers=2, out_dim=512)
    model = CLIPModel(img_enc, txt_enc, proj_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # run a couple of epochs
    epochs = 3
    for ep in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {ep+1}/{epochs} loss={loss:.4f}")

    # zero-shot example: classify an image with candidate prompts
    sample_img = images[0]
    class_prompts = ["a photo of a dog", "a photo of a cat", "a photo of a car"]
    probs, logits = zero_shot_classify(model, sample_img, class_prompts, vocab, device, max_len=16)
    print("Zero-shot logits:", logits)
    print("Zero-shot probs:", probs)


if __name__ == "__main__":
    demo_run()
