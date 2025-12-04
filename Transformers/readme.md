
---

## **I. Based on Attention Mechanism**

Transformers can differ mainly in **how attention is computed** — the heart of the model.

| **Category**                                  | **Type**                                                                                         | **Description / Formula Notes**                                                     | **Examples**                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------- |
| **1. Self-Attention Transformers**            | Encoder-only or full models where every token attends to every other token in the same sequence. | ( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V ) | BERT, ViT, GPT                          |
| **2. Cross-Attention Transformers**           | Decoder attends to encoder output, not to itself. Used in seq2seq models.                        | ( Q_{dec}K_{enc}^T ) cross-attention                                                | T5, BART                                |
| **3. Causal / Masked Self-Attention**         | Masked to prevent tokens from attending to future tokens (used in autoregressive models).        | Mask upper triangle of attention matrix                                             | GPT, LLaMA, Falcon                      |
| **4. Multi-Head Attention (MHA)**             | Multiple attention heads capture different subspace relations.                                   | ( \text{Concat}(head_1, ..., head_h)W^O )                                           | Almost all Transformers                 |
| **5. Sparse / Local Attention**               | Only attends to nearby tokens or structured subsets to reduce complexity.                        | Reduces O(n²) to O(n log n) or O(n)                                                 | Longformer, BigBird, Sparse Transformer |
| **6. Linearized / Low-Rank Attention**        | Uses kernel tricks or low-rank projection to approximate attention efficiently.                  | Linear attention via Φ(Q)Φ(K)ᵀ                                                      | Performer, Linformer, Nyströmformer     |
| **7. Rotary / Relative Positional Attention** | Adds position through rotation or relative encoding instead of absolute embedding.               | Uses RoPE or relative bias                                                          | LLaMA, Transformer-XL, DeBERTa          |
| **8. Mixture-of-Experts Attention**           | Routes tokens to expert attention heads.                                                         | Dynamic routing per token                                                           | GLaM, Switch Transformer                |
| **9. Memory-Augmented Attention**             | Adds external memory vectors attended over time.                                                 | ( [K; M], [V; M_V] )                                                                | Transformer-XL, Compressive Transformer |
| **10. State Space / Hybrid Attention**        | Combines recurrent or convolutional updates with attention.                                      | Continuous or discrete state equations                                              | Mamba, Hyena, RWKV                      |

---

## **II. Based on Model Architecture (Encoder/Decoder Structure)**

| **Architecture Type**             | **Description**                                                                                   | **Examples**                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **1. Encoder-only**               | Learns bidirectional context. Used for representation tasks (classification, feature extraction). | BERT, RoBERTa, DeBERTa, DistilBERT                     |
| **2. Decoder-only**               | Autoregressive. Used for text generation.                                                         | GPT-2/3/4, LLaMA, Falcon, Mistral                      |
| **3. Encoder–Decoder (Seq2Seq)**  | Encoder processes input → Decoder generates output conditioned on encoder states.                 | T5, BART, mT5, Marian, ProphetNet                      |
| **4. Dual-Encoder**               | Separate encoders for query and context; used in retrieval.                                       | CLIP (text–image), Sentence-BERT, DPR                  |
| **5. Multi-Modal Transformers**   | Combine multiple modalities (text, image, audio, video).                                          | CLIP, Flamingo, Kosmos, GPT-4V                         |
| **6. Vision Transformers (ViTs)** | Adapt transformer architecture to image patches.                                                  | ViT, Swin Transformer, DeiT                            |
| **7. Audio/Video Transformers**   | Apply temporal + spatial attention.                                                               | Audio Spectrogram Transformer, TimeSformer, Video Swin |

---

## **III. Based on Efficiency and Scaling Techniques**

| **Category**                              | **Approach**                               | **Representative Models**                           |
| ----------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| **1. Sparse Transformers**                | Attend to selected tokens.                 | Sparse Transformer, Longformer, BigBird             |
| **2. Low-Rank Transformers**              | Reduce attention matrix rank.              | Linformer, Performer                                |
| **3. Recurrent Memory Transformers**      | Extend context length via memory.          | Transformer-XL, Compressive Transformer             |
| **4. Retrieval-Augmented Transformers**   | Retrieve external documents for context.   | RETRO, RAG, Atlas                                   |
| **5. Quantized / Distilled Transformers** | Reduce parameter size for efficiency.      | DistilBERT, TinyBERT, MobileBERT                    |
| **6. Long-Context Transformers**          | Efficient handling of very long sequences. | Mistral, Longformer, Hyena, RWKV, Claude 3.5 Sonnet |

---

## **IV. Based on Application Domain**

| **Domain**                            | **Transformer Type**                       | **Examples**                 |
| ------------------------------------- | ------------------------------------------ | ---------------------------- |
| **Natural Language Processing (NLP)** | Encoder, Decoder, Seq2Seq                  | BERT, GPT, T5                |
| **Vision (CV)**                       | Vision Transformers                        | ViT, Swin, ConvNeXt-V2       |
| **Speech / Audio**                    | Spectrogram or waveform transformers       | Whisper, AST                 |
| **Multimodal (Text + Image/Video)**   | Cross-modal transformers                   | CLIP, Flamingo, Kosmos       |
| **Reinforcement Learning**            | Sequence modeling for trajectories         | Decision Transformer, Gato   |
| **Graph Data**                        | Graph Attention Networks (GAT), Graphormer | GAT, SAN                     |
| **Protein / Molecule Modeling**       | Structure–sequence transformers            | AlphaFold, ESM-2, ProtBERT   |
| **Code / Programming**                | Code understanding or generation           | CodeT5, CodeLlama, PolyCoder |

---

## **V. Emerging / Experimental Transformer Variants**

| **Variant Type**                              | **Key Innovation**                             | **Example Models**                |
| --------------------------------------------- | ---------------------------------------------- | --------------------------------- |
| **1. Mixture-of-Experts (MoE) Transformers**  | Dynamic routing across subnetworks             | GLaM, Switch Transformer, Mixtral |
| **2. Retrieval-Augmented Transformers (RAG)** | Add external retrieval context                 | RETRO, Atlas                      |
| **3. Sparse Mixture Models**                  | Combine sparsity and MoE                       | Mixtral, DeepSeekMoE              |
| **4. Continuous-Time Transformers**           | Replace discrete layers with ODE-like dynamics | Neural ODE Transformer            |
| **5. Memory-State Transformers**              | Persistent token memories                      | MEMIT, RecurrentGPT               |
| **6. Hybrid Attention + Convolution Models**  | Combine local conv + global attention          | CoAtNet, LeViT                    |

---

## **Summary Hierarchy**

```
Transformers
│
├── Based on Attention
│   ├── Self-Attention
│   ├── Cross-Attention
│   ├── Causal / Masked
│   ├── Sparse / Linear / Rotary / MoE / Memory
│
├── Based on Architecture
│   ├── Encoder-only
│   ├── Decoder-only
│   ├── Encoder–Decoder
│   ├── Dual / Multimodal / Vision / Audio
│
├── Based on Efficiency
│   ├── Sparse / Low-Rank / Retrieval / Memory
│
├── Based on Domain
│   ├── NLP / Vision / Audio / Multimodal / Graph / RL / Code
│
└── Emerging Variants
    ├── MoE / RAG / Continuous-Time / Hybrid
```

---

