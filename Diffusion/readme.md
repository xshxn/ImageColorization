# Cartoon Image Colorization using Diffusion Models (DDPM)

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to colorize black-and-white cartoon images. Unlike conditional GANs (like Pix2Pix), this approach uses a probabilistic diffusion process to iteratively refine noise into color, conditioned on the grayscale structure.

## Results

| Input (Grayscale) | Output (Generated Color) | Ground Truth |
| :---: | :---: | :---: |
| ![Input](path/to/input_gray.png) | ![Output](path/to/output_color.png) | ![Real](path/to/ground_truth.png) |

> *Note: The model demonstrates the ability to infer correct semantic colors (e.g., BMO is green, Finn's hat is white) solely from grayscale inputs.*

---

## Methodology

### 1. The Model: Conditional UNet
The core architecture is a **Medium-sized UNet** (modified for diffusion) that predicts noise residuals.
* **Conditioning:** The model is "guided" by the grayscale image. We concatenate the **Noisy Latent (3 channels)** and the **Grayscale Input (1 channel)** to create a 4-channel input tensor.
* **Time Embeddings:** Sinusoidal time embeddings are injected at every residual block to tell the model which timestep $t$ it is currently denoising.

### 2. The Noise Schedule
* **Training:** We used a standard forward diffusion process where Gaussian noise is added over $T=1000$ steps.
* **Scheduler:** We utilized a **Cosine Beta Schedule** (Nichol & Dhariwal, 2021) instead of a Linear schedule. This was critical for preventing "washed out" or grey-scale outputs, as it preserves information better in the middle steps of diffusion.

### 3. Sampling (Inference)
* **DDIM Sampling:** Instead of the standard Markovian sampling (which requires 1000 steps), we implemented **DDIM (Denoising Diffusion Implicit Models)**.
* This allows us to skip steps deterministically, generating high-quality images in just **50 steps** during inference.

---

## Technical Optimizations

To train this on limited hardware (Kaggle P100/T4 GPUs), several optimizations were implemented:

* **Mixed Precision Training (FP16):** Used `torch.cuda.amp` (GradScaler) to reduce VRAM usage and speed up training computations.
* **Gradient Accumulation:** Training with a batch size of 2 was necessary for memory, but unstable. We accumulated gradients over 4 steps to simulate an effective **Batch Size of 8**.
* **EMA (Exponential Moving Average):** Maintained a shadow copy of the model weights with a decay of 0.9999. This stabilized the final generation and reduced high-frequency artifacts (static).
* **Robust Data Loading:** Implemented safe handling for `PIL.Image.LOAD_TRUNCATED_IMAGES` to handle corrupt files in the large dataset without crashing the pipeline.

---

## Challenges & Solutions

During the development of this project, several significant issues were encountered and resolved:

1.  **The "Static" Problem:**
    * **Issue:** The initial inference outputs were pure static noise.
    * **Solution:** Identified a **Normalization Mismatch**. The model expected inputs scaled to `[-1, 1]`, but the dataloader was providing `[0, 1]`. Rescaling the inputs fixed the signal processing.

2.  **The "Deep Fried" Images:**
    * **Issue:** Colors were exploding into neon artifacts during inference.
    * **Solution:** Implemented **Dynamic Clipping** during sampling. By clamping the predicted $x_0$ to `[-1, 1]` at every step, we forced the pixel values to stay within a valid range, preventing mathematical snowballing.

3.  **The "Ghosting" Effect:**
    * **Issue:** The model generated the correct shape but the colors were dull/grey.
    * **Solution:** Switched from a **Linear** noise schedule to a **Cosine** schedule. The Linear schedule was destroying information too quickly; Cosine kept the signal intact longer, allowing the model to learn subtle color relationships better.

---

## How to Run

### Requirements
* Python 3.8+
* PyTorch
* Torchvision
* PIL, Matplotlib, Tqdm

### Inference
To run the model on a random image from the dataset:
```python
# Load the model and run inference with fixed scaling
python inference.py

Training
To retrain the model from scratch:

python train.py --epochs 12 --batch_size 2 --accumulate 4