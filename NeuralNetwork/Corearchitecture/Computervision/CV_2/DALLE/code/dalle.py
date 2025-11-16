"""
small_dalle_toy.py
A minimal, runnable toy DALL·E-style pipeline:
- VQ-VAE (encoder + quantizer + decoder)
- Autoregressive Transformer conditioned on text to predict image tokens
- Training pipeline: first VQ-VAE, then AR Transformer
- Sampling demo

Designed for clarity and to be runnable on small datasets / CPU.
"""

import math
import random
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

# ----------------------------
# Utilities
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def subsequent_mask(size: int, device):
    """Causal mask for self-attention: shape (size, size) with -inf for illegal positions."""
    attn_shape = (size, size)
    mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    return mask  # True values will be masked by transformer as needed


# ----------------------------
# Tiny dataset (toy data)
# ----------------------------
class ToyImageTextDataset(Dataset):
    """Creates small synthetic dataset: small images and short captions.
       For demo we create colored squares associated with captions.
    """
    def __init__(self, n_samples=500, img_size=64, vocab=None, max_text_len=8):
        super().__init__()
        self.n_samples = n_samples
        self.img_size = img_size
        self.max_text_len = max_text_len

        # generate synthetic images and captions (simple shapes/solid colors)
        choices = ["red square", "green circle", "blue triangle", "yellow square", "purple circle"]
        self.captions = [random.choice(choices) for _ in range(n_samples)]
        self.images = [self._make_image_from_caption(c) for c in self.captions]

        # tokenization
        if vocab is None:
            self.vocab = self.build_vocab(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = self.images[idx]  # tensor [3,H,W]
        cap = self.captions[idx]
        toks = torch.tensor(self.vocab.tokenize(cap, max_len=self.max_text_len), dtype=torch.long)
        return img, toks

    def _make_image_from_caption(self, caption: str):
        # Very simple synthetic images: colored square at center or circle using channels
        img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        color = torch.zeros(3)
        if "red" in caption:
            color = torch.tensor([1.0, 0.0, 0.0])
        elif "green" in caption:
            color = torch.tensor([0.0, 1.0, 0.0])
        elif "blue" in caption:
            color = torch.tensor([0.0, 0.0, 1.0])
        elif "yellow" in caption:
            color = torch.tensor([1.0, 1.0, 0.0])
        elif "purple" in caption:
            color = torch.tensor([0.6, 0.0, 0.6])
        # shape intensity
        shape = "square"
        if "circle" in caption:
            shape = "circle"
        elif "triangle" in caption:
            shape = "triangle"
        H = self.img_size
        c = H // 2
        r = H // 4
        for y in range(H):
            for x in range(H):
                if shape == "square":
                    if (abs(y - c) <= r) and (abs(x - c) <= r):
                        img[:, y, x] = color
                elif shape == "circle":
                    if (y - c) ** 2 + (x - c) ** 2 <= r * r:
                        img[:, y, x] = color
                elif shape == "triangle":
                    # simple isosceles triangle
                    if y >= c - r and abs(x - c) <= (y - (c - r)):
                        img[:, y, x] = color
        # add small gaussian noise
        img = img + 0.02 * torch.randn_like(img)
        img = img.clamp(0.0, 1.0)
        return img

    @staticmethod
    def build_vocab(captions):
        # simple whitespace vocab
        toks = {}
        toks["<pad>"] = 0
        toks["<unk>"] = 1
        for c in captions:
            for t in c.strip().lower().split():
                if t not in toks:
                    toks[t] = len(toks)
        return SimpleVocab(toks)

# ----------------------------
# Simple whitespace vocab helper
# ----------------------------
class SimpleVocab:
    def __init__(self, token2idx=None):
        if token2idx is None:
            token2idx = {"<pad>": 0, "<unk>": 1}
        self.token2idx = token2idx
        self.idx2token = {v:k for k,v in token2idx.items()}

    def tokenize(self, text: str, max_len=8):
        toks = []
        for w in text.strip().lower().split():
            toks.append(self.token2idx.get(w, self.token2idx["<unk>"]))
            if len(toks) >= max_len:
                break
        if len(toks) < max_len:
            toks += [self.token2idx["<pad>"]] * (max_len - len(toks))
        return toks

    def __len__(self):
        return len(self.token2idx)

# ----------------------------
# VQ-VAE: Encoder, Quantizer, Decoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=64, latent_dim=64, downsample=4):
        super().__init__()
        # For 64x64 -> downsample by factor 'downsample' (e.g., 4 -> 16x16 -> latent grid 4x4 if conv stride)
        # We'll use conv layers to reduce spatial dims.
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1),  # /2
            nn.ReLU(),
            nn.Conv2d(hidden, hidden*2, 4, 2, 1),  # /4
            nn.ReLU(),
            nn.Conv2d(hidden*2, latent_dim, 3, 1, 1),  # keep spatial
        )

    def forward(self, x):
        return self.net(x)  # shape (B, latent_dim, H/4, W/4)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden=64, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden*2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden*2, hidden, 4, 2, 1),  # *2
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_ch, 4, 2, 1),  # *2
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    """A simple vector quantizer (VQ-VAE) with straight-through estimator."""
    def __init__(self, num_codes=64, code_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B, D, H, W) continuous latents
        returns:
          z_q: quantized latents (B, D, H, W)
          indices: (B, H, W) long indices
          loss_vq: codebook + commitment loss
        """
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat = flat.view(-1, D)  # (B*H*W, D)

        # compute distances
        # embedding: (K, D)
        emb = self.embedding.weight  # (K, D)
        # distances: (N, K)
        distances = (flat.pow(2).sum(1, keepdim=True)
                     - 2 * flat @ emb.t()
                     + emb.pow(2).sum(1).unsqueeze(0))

        indices = torch.argmin(distances, dim=1)  # (N,)
        z_q = emb[indices].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        # compute loss
        # stop gradient on indices embedding path
        loss_vq = F.mse_loss(z_q.detach(), z_e) + self.commitment_cost * F.mse_loss(z_q, z_e.detach())
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        indices = indices.view(B, H, W)
        return z_q_st, indices, loss_vq


class VQVAE(nn.Module):
    def __init__(self, img_size=64, in_ch=3, hidden=64, latent_dim=64, n_codes=64):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, hidden=hidden, latent_dim=latent_dim)
        self.quant = VectorQuantizer(num_codes=n_codes, code_dim=latent_dim, commitment_cost=0.25)
        self.decoder = Decoder(latent_dim=latent_dim, hidden=hidden, out_ch=in_ch)

    def forward(self, x):
        z_e = self.encoder(x)  # (B, D, H', W')
        z_q, indices, vq_loss = self.quant(z_e)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss

    def encode_to_indices(self, x):
        with torch.no_grad():
            z_e = self.encoder(x)
            # use quantizer distances without straight-through for indices
            B, D, H, W = z_e.shape
            flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
            emb = self.quant.embedding.weight  # (K, D)
            distances = (flat.pow(2).sum(1, keepdim=True)
                         - 2 * flat @ emb.t()
                         + emb.pow(2).sum(1).unsqueeze(0))
            indices = torch.argmin(distances, dim=1).view(B, H, W)
            return indices  # (B, H, W)

    def decode_from_indices(self, indices):
        # indices: (B, H, W) -> map to embedding and decode
        emb = self.quant.embedding.weight  # (K, D)
        B, H, W = indices.shape
        z_q = emb[indices.view(-1)].view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        recon = self.decoder(z_q)
        return recon


# ----------------------------
# Autoregressive Transformer (simple)
# ----------------------------
class ARTransformer(nn.Module):
    """
    Autoregressive Transformer that predicts next token in the concatenated sequence:
    [ text_tokens (L) , image_tokens (M) ]
    We compute logits only for image token positions and apply cross-entropy there.
    """
    def __init__(self, n_text_vocab, n_image_codes, d_model=128, n_head=4, n_layers=4, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_text_vocab = n_text_vocab
        self.n_image_codes = n_image_codes
        self.max_len = max_len

        # embeddings
        self.text_embed = nn.Embedding(n_text_vocab, d_model)
        self.image_embed = nn.Embedding(n_image_codes, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)

        # transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4*d_model, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # output head: predict image code logits
        self.to_logits = nn.Linear(d_model, n_image_codes)

    def forward(self, text_tokens, image_tokens):
        """
        text_tokens: (B, L_text)
        image_tokens: (B, M)  -- this is the target sequence but we feed full sequence with teacher forcing
        We'll construct input sequence S = [text, image] and return logits for positions corresponding to image tokens predicting next token.
        """
        B, L = text_tokens.shape
        _, M = image_tokens.shape
        total_len = L + M
        assert total_len <= self.max_len, "increase max_len"

        # embeddings
        text_e = self.text_embed(text_tokens)  # (B, L, d)
        image_e = self.image_embed(image_tokens)  # (B, M, d)  (these are input tokens for teacher forcing)
        x = torch.cat([text_e, image_e], dim=1)  # (B, L+M, d)
        x = x + self.pos_embed[:, :total_len, :]

        # shape for transformer: (S, B, d)
        x = x.transpose(0, 1)  # (S, B, d)
        # causal mask (S,S)
        mask = subsequent_mask(total_len, device=x.device)
        # transformer expects mask where True indicates positions to be masked
        out = self.transformer(x, mask=mask)  # (S, B, d)
        out = out.transpose(0, 1)  # (B, S, d)

        # logits for entire sequence -> but we only compute loss on image-prediction positions:
        logits = self.to_logits(out)  # (B, S, n_image_codes)
        # Extract logits for positions that correspond to image tokens (these logits predict the token at that same position)
        logits_image_positions = logits[:, L:, :]  # (B, M, K)
        return logits_image_positions  # model predicts token at positions L..L+M-1

    def sample(self, text_tokens, M, temperature=1.0, top_k=None):
        """Autoregressive sampling: given text tokens (B, L), produce M image tokens."""
        self.eval()
        B, L = text_tokens.shape
        samples = torch.full((B, M), fill_value=0, dtype=torch.long, device=text_tokens.device)  # start with zeros
        with torch.no_grad():
            for i in range(M):
                # feed current prefix (text + sampled image tokens up to i)
                cur_image_prefix = samples.clone()
                # for positions > i we keep zeros — but mask prevents seeing future (we'll still pass full M but future positions logits won't be used)
                logits = self.forward(text_tokens, cur_image_prefix)  # (B, M, K)
                # take logits for position i
                logits_i = logits[:, i, :] / max(1e-6, temperature)
                if top_k is not None:
                    top_k_val = min(top_k, logits_i.size(-1))
                    v, _ = logits_i.topk(top_k_val)
                    min_v = v[:, -1].unsqueeze(-1)
                    logits_i = torch.where(logits_i < min_v, torch.full_like(logits_i, -1e9), logits_i)
                probs = F.softmax(logits_i, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                samples[:, i] = next_tokens
        self.train()
        return samples  # (B, M)


# ----------------------------
# Training helpers
# ----------------------------
def train_vqvae(vqvae: VQVAE, dataloader, n_epochs=10, lr=2e-4, log_dir="logs"):
    optim = torch.optim.Adam(vqvae.parameters(), lr=lr)
    vqvae.to(device)
    os.makedirs(log_dir, exist_ok=True)
    for epoch in range(1, n_epochs + 1):
        vqvae.train()
        total_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            recon, indices, vq_loss = vqvae(imgs)
            rec_loss = F.mse_loss(recon, imgs)
            loss = rec_loss + vq_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(dataloader.dataset)
        print(f"[VQ-VAE] Epoch {epoch}/{n_epochs} avg_loss={avg:.6f}")
        # save sample reconstructions
        with torch.no_grad():
            vqvae.eval()
            sample_imgs = dataloader.dataset.images[:8]
            sample_imgs = torch.stack(sample_imgs).to(device)
            recon, _, _ = vqvae(sample_imgs)
            grid = make_grid(torch.cat([sample_imgs, recon], dim=0).cpu(), nrow=8)
            save_image(grid, os.path.join(log_dir, f"vq_recon_epoch{epoch}.png"))
    return vqvae


def prepare_token_dataset(vqvae: VQVAE, dataset: ToyImageTextDataset):
    """Encode whole dataset images to code indices and return arrays."""
    vqvae.to(device)
    vqvae.eval()
    imgs = torch.stack(dataset.images).to(device)
    with torch.no_grad():
        indices = vqvae.encode_to_indices(imgs)  # (N, H', W')
    # flatten spatial to sequence
    N, Hq, Wq = indices.shape
    M = Hq * Wq
    sequences = indices.view(N, M).cpu()
    # return paired text tokens and image token sequences
    texts = []
    for c in dataset.captions:
        texts.append(torch.tensor(dataset.vocab.tokenize(c, max_len=dataset.max_text_len), dtype=torch.long))
    texts = torch.stack(texts)  # (N, L)
    return texts, sequences, (Hq, Wq)


def train_ar_transformer(transformer: ARTransformer, texts, image_sequences, dataset, n_epochs=20, lr=3e-4, batch_size=32, log_dir="logs"):
    # texts: (N, L), image_sequences: (N, M)
    transformer.to(device)
    opt = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=0.01)
    N = texts.size(0)
    indices = list(range(N))
    os.makedirs(log_dir, exist_ok=True)
    for epoch in range(1, n_epochs+1):
        random.shuffle(indices)
        transformer.train()
        total_loss = 0.0
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            b_texts = texts[batch_idx].to(device)
            b_imgs = image_sequences[batch_idx].to(device)
            logits = transformer(b_texts, b_imgs)  # (B, M, K)
            # targets are the actual image tokens at those positions
            targets = b_imgs  # (B, M)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * b_texts.size(0)
        avg = total_loss / N
        print(f"[AR] Epoch {epoch}/{n_epochs} avg_loss={avg:.6f}")
    return transformer


# ----------------------------
# Sampling helper
# ----------------------------
def sample_and_save(transformer: ARTransformer, vqvae: VQVAE, vocab: SimpleVocab, prompt: str, HqWq: Tuple[int,int], max_text_len=8, top_k=None, out_file="sample.png"):
    transformer.to(device)
    vqvae.to(device)
    # tokenize prompt
    tok = torch.tensor([vocab.tokenize(prompt, max_len=max_text_len)], dtype=torch.long).to(device)
    Hq, Wq = HqWq
    M = Hq * Wq
    sampled_indices = transformer.sample(tok, M, temperature=1.0, top_k=top_k)  # (1, M)
    # reshape to (1, Hq, Wq)
    idx_grid = sampled_indices.view(1, Hq, Wq)
    # decode
    recon = vqvae.decode_from_indices(idx_grid.to(device))
    save_image(recon.clamp(0,1).cpu(), out_file)
    print(f"Saved sample to {out_file}")


# ----------------------------
# Demo / main
# ----------------------------
def main_demo():
    # Hyperparams (small)
    img_size = 64
    n_samples = 400
    batch = 32
    max_text_len = 8

    # Build dataset
    dataset = ToyImageTextDataset(n_samples=n_samples, img_size=img_size, max_text_len=max_text_len)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    # VQ-VAE small
    n_codes = 64
    latent_dim = 64
    vqvae = VQVAE(img_size=img_size, in_ch=3, hidden=32, latent_dim=latent_dim, n_codes=n_codes)

    print("Training VQ-VAE...")
    vqvae = train_vqvae(vqvae, dataloader, n_epochs=8, lr=2e-4, log_dir="logs")

    # Prepare tokens dataset
    print("Encoding dataset to tokens...")
    texts, sequences, (Hq, Wq) = prepare_token_dataset(vqvae, dataset)
    print(f"Token grid: {Hq}x{Wq} -> sequence length {Hq*Wq}")

    # AR transformer: sizes
    n_text_vocab = len(dataset.vocab)
    transformer = ARTransformer(n_text_vocab=n_text_vocab, n_image_codes=n_codes, d_model=128, n_head=4, n_layers=4, max_len=max_text_len + Hq*Wq)

    print("Training AR transformer...")
    transformer = train_ar_transformer(transformer, texts, sequences, dataset, n_epochs=30, lr=3e-4, batch_size=32, log_dir="logs")

    # Sample examples
    prompts = ["red square", "green circle", "blue triangle"]
    for i, p in enumerate(prompts):
        sample_and_save(transformer, vqvae, dataset.vocab, p, (Hq, Wq), max_text_len=max_text_len, top_k=32, out_file=f"sample_{i}_{p.replace(' ', '_')}.png")

    print("Demo finished. Samples saved to current directory.")

if __name__ == "__main__":
    main_demo()
