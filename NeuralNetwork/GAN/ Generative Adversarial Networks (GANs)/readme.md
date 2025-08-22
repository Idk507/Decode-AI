## **1. Generative Adversarial Networks (GANs)**
GANs consist of a **generator** and a **discriminator** competing in a minimax game. Below are key GAN variants:

### **Foundational GANs**
1. **Vanilla GAN (2014)** – Original GAN by Ian Goodfellow.
2. **DCGAN (2015)** – Deep Convolutional GAN, introduced CNNs for stable training.
3. **Conditional GAN (cGAN) (2014)** – Adds conditional input (e.g., class labels) to both generator and discriminator.

### **Improved Training & Stability**
4. **Wasserstein GAN (WGAN) (2017)** – Uses Wasserstein distance for better training.
5. **WGAN-GP (2017)** – Improves WGAN with gradient penalty.
6. **Least Squares GAN (LSGAN) (2016)** – Uses least squares loss to stabilize training.
7. **Spectral Normalization GAN (SNGAN) (2018)** – Applies spectral normalization to stabilize training.

### **Architectural Advances**
8. **ProGAN (2017)** – Progressive growing of GANs for high-res images.
9. **StyleGAN (2018) & StyleGAN2 (2020)** – Introduces style-based generation for high-quality faces.
10. **BigGAN (2018)** – Large-scale GAN for high-fidelity generation.
11. **Self-Attention GAN (SAGAN) (2018)** – Uses self-attention for global dependencies.

### **Specialized GANs**
12. **CycleGAN (2017)** – Unpaired image-to-image translation.
13. **DiscoGAN (2017)** – Similar to CycleGAN for cross-domain translation.
14. **StarGAN (2018)** – Multi-domain image translation.
15. **InfoGAN (2016)** – Unsupervised disentangled representation learning.
16. **SinGAN (2019)** – Learns from a single image for diverse generation.

### **Video & 3D GANs**
17. **VGAN (2016)** – Video GAN for spatio-temporal generation.
18. **MoCoGAN (2018)** – Motion and content decomposition for video generation.
19. **3D-GAN (2016)** – Generates 3D objects.
