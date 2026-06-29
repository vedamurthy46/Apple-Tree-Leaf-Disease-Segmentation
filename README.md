# Apple Leaf Disease Segmentation (ALDS)
Binary semantic segmentation of apple leaf disease lesions using a custom prototype-guided UNet — **PSPE-Net**.

## Results

| Metric | Score |
|--------|-------|
| IoU | 97.69 % |
| Dice Score | 98.83 % |

Evaluated on the held-out test split (20 % of the dataset) at a classification threshold of 0.5.

## Model Architecture — PSPE-UNet

PSPE-Net extends the standard UNet with a learnable prototype-similarity classification head in place of a plain 1×1 output convolution.

| Component | Details |
|-----------|---------|
| **Encoder** | 4 × ConvBlock (3 → 64 → 128 → 256 → 512) + bottleneck (1 024 ch), MaxPool2d ×2 downsampling |
| **Decoder** | 4 × ConvTranspose2d upsampling with skip connections from corresponding encoder stages |
| **Projection Head** | 1×1 Conv2d → L2-normalised 64-dim embedding per pixel |
| **Prototype Similarity** | 2 learnable prototypes (background / lesion); cosine similarity + softmax (temperature = 10.0); outputs lesion probability map |

Each **ConvBlock** follows the pattern:  
`Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d → ReLU`

## Dataset

**Apple Leaf Disease Segmentation Dataset (ATLDSD)**  
Platform: Kaggle — `apple-leaf-disease/ATLDSD`

Expected directory layout:

```
ATLDSD/
  <class_name>/
    image/    ← RGB leaf images
    label/    ← binary PNG masks  (lesion = white, background = black)
```

## Data Split

| Split | Proportion |
|-------|-----------|
| Train | 80 % |
| Test | 20 % |

No separate validation set is used. Splits are generated with `torch.utils.data.random_split` using a fixed generator seed, making them **exactly reproducible** without needing saved index files:

python
generator = torch.Generator().manual_seed(42)
train_indices, test_indices = random_split(
    range(len(full_ds)), [train_size, test_size], generator=generator
)

Re-running the notebook with `seed = 42` will produce the identical partition.

## Preprocessing

Applied to **all** splits (train and test):

| Step | Detail |
|------|--------|
| Resize | 256 × 256 pixels (NEAREST interpolation for masks) |
| To Tensor | `torchvision.transforms.functional.to_tensor` |
| Normalise (images only) | Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]` (ImageNet) |
| Binarise (masks only) | Pixel > 0 → 1.0 (lesion), else 0.0 (background) |

## Augmentation

Applied to the **training split only**. The test split receives no augmentation beyond preprocessing.

| Transform | Parameters |
|-----------|-----------|
| Random Horizontal Flip | p = 0.5 |
| Random Vertical Flip | p = 0.5 |
| Random Rotation | angle ∈ [−30°, +30°] (uniform) |
| Color Jitter (image only) | brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1 |

Color jitter is applied only to the image; the corresponding mask is not colour-transformed.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| LR scheduler | ReduceLROnPlateau |
| Scheduler mode | `min` (monitors test loss) |
| Scheduler patience | 5 epochs |
| Scheduler factor | 0.5 |
| Loss | BCE + Dice |
| Dice smooth | 1.0 |
| Embedding dim | 64 |
| Prototype temperature | 10.0 |
| Segmentation threshold | 0.5 |
| Random seed (split) | 42 |
| DataLoader workers | 2 |


## Loss Function
L_total = L_BCE(pred, target) + L_Dice(pred, target)
L_Dice = 1 − (2 · Σ(pred · target) + smooth) / (Σpred + Σtarget + smooth)


`smooth = 1.0` is applied to both numerator and denominator to avoid division by zero.

## Evaluation Metrics

Both metrics are computed after binarising predictions at threshold = 0.5.  
A small epsilon (ε = 1 × 10⁻⁸) is added to numerator and denominator to avoid zero-division on empty masks.

**IoU (Intersection over Union)**

IoU = (|pred ∩ target| + ε) / (|pred ∪ target| + ε)

**Dice Score**

Dice = (2 · |pred ∩ target| + ε) / (|pred| + |target| + ε)

## Training Loop

- The model is trained for 50 epochs.
- Per-epoch train and test loss, IoU, and Dice scores are logged and plotted at the end.
- The best checkpoint (highest test IoU) is saved as `best_model.pth` via:

```python
torch.save(model.state_dict(), 'best_model.pth')
```

- To reload for inference:

```python
model = PSPE_UNet()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## Reproducibility

The train/test split is fully reproducible using `seed = 42` as shown in the **Data Split** section above.

For bit-level determinism across all random operations in a run, add the following at the top of the notebook:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

> Note: `torch.backends.cudnn.deterministic = True` may reduce GPU throughput slightly but ensures identical outputs across runs.

## Environment

| Component | Version / Spec |
|-----------|---------------|
| Python | 3.14.2 |
| PyTorch | 2.10.0 |
| TensorFlow | 2.21 |
| CUDA | 12.6 |
| cuDNN | 9.10.2.21 |
| OS | Windows 10 Pro |
| CPU | Intel Core i7-7700K @ 4.20 GHz |
| GPU | NVIDIA GeForce RTX 3050 8 GB |
| RAM | 32 GB |
| Storage | 1.6 TB SSD |

### Install core dependencies

```bash
pip install torch torchvision pillow matplotlib
```

## Usage

1. Download the ATLDSD dataset from Kaggle and place it at:
   ```
   /kaggle/input/apple-leaf-disease/ATLDSD
   ```
   Or update the `root` variable at the top of the notebook to your local path.

2. Open `Veda-murthy-ALDS.ipynb` in Jupyter Notebook / JupyterLab / Kaggle.

3. Run all cells top-to-bottom. The notebook will:
   - Count and display sample images with their ground-truth masks
   - Build reproducible 80/20 train/test splits (seed = 42)
   - Instantiate PSPE-UNet and print parameter count
   - Train for 50 epochs, printing per-epoch metrics
   - Save the best checkpoint to `best_model.pth`
   - Plot Loss, IoU, and Dice training curves

## Repository Structure
```
.
├── Veda-murthy-ALDS.ipynb   # Full training, evaluation, and visualisation code
├── best_model.pth           # Best checkpoint (generated after training)
├── README.md
└── LICENSE
```
## License

This project is released under the [MIT License](LICENSE).
