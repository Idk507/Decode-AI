
---

### **What is a Warmup Strategy?**

A **warmup strategy** gradually increases the learning rate from a very small value (or zero) to a target value over a set number of steps or epochs at the start of training. After the warmup phase, the learning rate may follow a scheduling strategy (e.g., constant, decay, or cosine annealing). Think of it like warming up before a workout: you start slow to prepare the model for stable and effective training.

---

### **Why Use Warmup?**

Warmup is particularly important for **Transformers** (e.g., BERT, GPT) because:
- **Stabilizes training**: Transformers have complex architectures with many parameters and non-linearities. A high initial learning rate can cause unstable gradients, leading to divergence or poor convergence.
- **Handles large models**: Large models are sensitive to initialization, and warmup helps the model adjust weights gradually.
- **Improves optimization**: Early in training, the model’s weights are far from optimal. A small learning rate allows careful exploration of the loss landscape before taking larger steps.
- **Mitigates vanishing/exploding gradients**: Gradual learning rate increases prevent extreme updates that could destabilize training.

**When to use**: Warmup is standard in Transformer training (e.g., NLP tasks like machine translation, text generation) and other deep models where stability is a concern. It’s often paired with a decay schedule (e.g., linear or cosine decay) after the warmup phase.

---

### **How Does Warmup Work? (The Math)**


<img width="1065" height="694" alt="image" src="https://github.com/user-attachments/assets/704c363b-1e36-4c6f-844e-02eb1519ef38" />

**Example (Linear Warmup)**:


<img width="718" height="227" alt="image" src="https://github.com/user-attachments/assets/caee521e-b149-4f83-99bc-ef861ceae025" />

---

### **Warmup in Transformers**

In Transformers, warmup is critical due to:
- **Large parameter count**: Models like BERT or GPT have millions/billions of parameters, making them sensitive to large initial updates.
- **Attention mechanisms**: Self-attention layers can amplify small changes, so warmup stabilizes early training.
- **Pre-training and fine-tuning**: Warmup is used in both phases, especially in pre-training on large datasets (e.g., language modeling).

A common practice (from the Transformer paper) is to use linear warmup followed by inverse square root decay, as shown above. This is often called the **Noam schedule**.

---

### **Types of Warmup Strategies**

1. **Linear Warmup with Constant**: Increase linearly to \( \eta_{\text{max}} \), then keep it constant.
2. **Linear Warmup with Decay**: Increase linearly, then decay (e.g., linearly, exponentially, or inverse square root).
3. **Noam Schedule (Transformer)**: Linear warmup with inverse square root decay, tailored for Transformers.
4. **Cosine Warmup**: Increase learning rate following a cosine curve, then transition to cosine annealing.

---

### **Code Example in PyTorch**

Below is a PyTorch example implementing a **Transformer model** with a **linear warmup followed by inverse square root decay** (Noam schedule) for a toy sequence-to-sequence task (e.g., translating numbers to their string representation). I’ll also show how to combine it with cosine annealing for comparison.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset

# 1. Custom Warmup Scheduler (Noam Schedule)
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 2. Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc(output)

# 3. Toy Dataset (Numbers to Strings, e.g., 123 -> "one two three")
class NumberDataset(Dataset):
    def __init__(self, size=1000):
        self.data = [(str(i), " ".join([["zero", "one", "two", "three", "four"][int(d)] for d in str(i)])) for i in range(size)]
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "zero": 3, "one": 4, "two": 5, "three": 6, "four": 7}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = [self.vocab[c] for c in src]
        tgt_ids = [self.vocab["<sos>"]] + [self.vocab[w] for w in tgt.split()] + [self.vocab["<eos>"]]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# 4. Training Setup
dataset = NumberDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = SimpleTransformer(vocab_size=8, d_model=128, nhead=4, num_layers=2)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=0.0)  # LR set by scheduler

# 5. Scheduler (Noam or CosineAnnealing with Warmup)
scheduler_type = "noam"  # Change to "cosine" to try cosine annealing
if scheduler_type == "noam":
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=4000, factor=1.0)
else:
    # Linear warmup + Cosine Annealing
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=4000),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
        ],
        milestones=[4000]
    )

# 6. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        tgt_input = tgt[:, :-1]  # Exclude <eos> for input
        tgt_output = tgt[:, 1:]  # Exclude <sos> for target
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        current_lr = scheduler.step() if scheduler_type == "noam" else scheduler.step()
        if scheduler_type != "noam":
            current_lr = scheduler.get_last_lr()[0]
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Learning Rate: {current_lr:.6f}")

print("Training finished!")
```

---

### **Explanation of the Code**

1. **Noam Scheduler**:
   - Implements the Transformer’s warmup + inverse square root decay.
   - Formula:<img width="449" height="58" alt="image" src="https://github.com/user-attachments/assets/36c7ee37-6d4e-42aa-96b6-c7a3d4be5ab5" />

   - `d_model=128` (embedding size), `warmup_steps=4000`, `factor=1.0` for scaling.

2. **Model**:
   - A simple Transformer for sequence-to-sequence (numbers to strings, e.g., “123” → “one two three”).
   - Uses `nn.Transformer` with 2 layers and 4 attention heads.

3. **Dataset**:
   - A toy dataset mapping numbers (0–999) to their word representations.
   - Vocabulary includes `<pad>`, `<sos>`, `<eos>`, and words like “zero”, “one”, etc.

4. **Scheduler Options**:
   - **Noam**: Linear warmup for 4000 steps, then inverse square root decay.
   - **Cosine Annealing with Warmup**: Linear warmup for 4000 steps (using `LinearLR`), followed by cosine annealing (`CosineAnnealingLR`).
   - Switch between them by changing `scheduler_type`.

5. **Training Loop**:
   - Processes batches, computes loss, updates weights, and steps the scheduler.
   - Prints epoch, loss, and current learning rate.

---

### **Why This Example?**
- **Task**: The number-to-string task is simple but demonstrates sequence-to-sequence learning, common in Transformer applications.
- **Noam Scheduler**: Widely used in Transformer training (e.g., BERT, GPT), making it relevant.
- **Cosine Alternative**: Shows how warmup integrates with other schedules like cosine annealing.
- **Warmup Steps**: 4000 steps is typical for Transformers, balancing stability and speed.

---

### **Visualizing the Warmup Schedule**

Here’s a chart showing the learning rate for the Noam scheduler and linear warmup + cosine annealing over 10,000 steps.

<img width="974" height="561" alt="image" src="https://github.com/user-attachments/assets/b9c66fc9-ea27-49fa-8028-fb9afbf2b7d0" />


**Chart Explanation**:
- **Noam Scheduler** (blue): Linearly increases to a peak at 4000 steps, then decays as $\( t^{-0.5} \).$
- **Linear Warmup + Cosine** (orange): Linearly increases to 0.001 at 4000 steps, then follows a cosine decay.

---

### **When to Use Warmup Strategies?**
- **Noam Scheduler**: Use for Transformers in NLP tasks (e.g., machine translation, text generation). It’s the default in many Transformer implementations.
- **Linear Warmup + Decay**: Use for general deep learning tasks or when combining with schedules like cosine annealing.
- **Cosine Warmup**: Use for smooth transitions in both warmup and decay phases, especially in fine-tuning.

---

### **Tips for Warmup**
1. **Choose Warmup Steps**: Set $\( T_{\text{warmup}} \) to 5–10% of total training steps (e.g., 4000 steps for 40,000 total steps).$
2. **Tune Peak LR**: For Noam, adjust `factor` and `d_model`. For linear warmup, tune $\( \eta_{\text{max}} \).$
3. **Combine with Decay**: Warmup is most effective when followed by a decay schedule (e.g., inverse square root, cosine, or linear).
4. **Monitor Stability**: If training diverges early, increase warmup steps or lower the peak learning rate.

---

### **Conclusion**
Warmup strategies are essential for stabilizing Transformer training by gradually increasing the learning rate early on. The Noam scheduler (linear warmup + inverse square root decay) is a go-to for Transformers, while linear warmup with cosine annealing is versatile for other models. The PyTorch example shows how to implement these for a sequence-to-sequence task, and the chart visualizes their behavior. If you want to explore a specific warmup strategy, a different dataset, or another framework (e.g., TensorFlow), let me know!
