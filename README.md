# Document Image Segmentation using U-Net on the PRImA Layout Analysis Dataset

A complete PyTorch pipeline for **multi-class semantic segmentation of document images** using **only U-Net**, followed by extraction of **class-specific deep representative feature vectors** from the trained network.

## Project Objective

The project takes a document image and its PAGE XML annotation, converts the XML regions into a pixel-wise multi-class mask, trains a U-Net to segment every supported PRImA region class, evaluates the segmentation, and then uses the trained U-Net bottleneck features to build a representative vector for each document/class pair.

The current 11-class setup is:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | TextRegion |
| 2 | MathsRegion |
| 3 | TableRegion |
| 4 | ImageRegion |
| 5 | GraphicRegion |
| 6 | LineDrawingRegion |
| 7 | ChartRegion |
| 8 | SeparatorRegion |
| 9 | NoiseRegion |
| 10 | FrameRegion |

## Pipeline

**PRImA images + PAGE XML → XML parsing → pixel-wise masks → resize/normalize → train/validation/test split → U-Net → segmentation logits → IoU/Dice evaluation → bottleneck feature extraction → class-specific vectors → class prototypes → PCA/t-SNE analysis**

## Key Features

- Multi-class semantic segmentation
- All available PRImA region classes
- U-Net architecture only
- PAGE XML to pixel-mask conversion
- Training/validation/test pipeline
- Cross-Entropy + Dice loss
- AdamW optimization
- Per-class IoU and Dice
- Pixel accuracy
- Mean IoU and Mean Dice
- Segmentation prediction export
- Deep feature extraction from the U-Net bottleneck
- Class-specific feature vectors
- Ground-truth and predicted-mask representations
- Class prototype generation
- PCA visualization
- Modular PyTorch implementation

## Dataset

The project uses the **PRImA Layout Analysis Dataset**, containing document images together with PAGE XML layout annotations.

Expected structure:

```text
datasets/
└── PRImA_Layout_Analysis_Dataset/
    ├── Images/
    └── XML/
```

The image and XML files should have matching names/stems:

```text
Images/page001.png
XML/page001.xml
```

Nested directories are also supported.

## Project Structure

```text
document_segmentation_prima/
│
├── configs/
│   ├── __init__.py
│   └── config.py
│
├── datasets/
│   └── PRImA_Layout_Analysis_Dataset/
│       ├── Images/
│       └── XML/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
│
├── results/
│   ├── checkpoints/
│   ├── predictions/
│   ├── feature_vectors/
│   └── plots/
│
├── train.py
├── evaluate.py
├── extract_features.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repository-url>
cd document_segmentation_prima

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For CUDA systems, install the appropriate PyTorch CUDA build for the target machine.

## Configuration

The main configuration is located at:

```text
configs/config.py
```

Important parameters include:

```python
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
```

The class definitions are centralized in `config.py` so that the same class IDs are used throughout mask generation, U-Net output channels, evaluation, prediction, and feature extraction.

## Training

Run:

```bash
python train.py
```

The training pipeline performs:

1. Image/XML pairing
2. Train/validation/test splitting
3. PAGE XML parsing
4. Pixel-wise mask generation
5. Image and mask resizing
6. Image normalization
7. Training augmentation
8. U-Net training
9. Cross-Entropy + Dice loss calculation
10. AdamW optimization
11. Validation
12. Learning-rate scheduling
13. Best-model checkpointing

The best model is saved to:

```text
results/checkpoints/best.pt
```

## Model Architecture

The project uses a standard U-Net encoder-decoder architecture.

```text
                    INPUT IMAGE
                         │
                         ▼
                  ┌─────────────┐
                  │   Encoder   │
                  │ Conv + Pool │
                  └──────┬──────┘
                         │
                    Downsampling
                         │
                         ▼
                  ┌───────────────┐
                  │  Bottleneck   │
                  │ 1024 channels │
                  └──────┬────────┘
                         │
                    Upsampling
                         │
                         ▼
                  ┌─────────────┐
                  │   Decoder   │
                  │ Conv + Up   │
                  └──────┬──────┘
                         │
                    Skip Connections
                         │
                         ▼
                 ┌────────────────┐
                 │ 1×1 Conv Head  │
                 └───────┬────────┘
                         │
                         ▼
                 11-Class Segmentation
```

The U-Net supports two modes:

```python
logits = model(image)
```

for normal segmentation, and:

```python
logits, bottleneck = model(
    image,
    return_features=True
)
```

for extracting deep representations.

## Evaluation

Run:

```bash
python evaluate.py
```

The following metrics are calculated:

- IoU for every class
- Dice coefficient for every class
- Pixel accuracy
- Mean IoU
- Mean Dice
- Foreground mean IoU
- Foreground mean Dice

Predicted masks are saved as:

```text
results/predictions/
```

and numerical results are written to:

```text
results/test_metrics.csv
```

## Class-Specific Deep Feature Extraction

The second major objective of the project is to obtain a deep feature representation for each document-layout class.

Run:

```bash
python extract_features.py
```

The process is:

```text
Document Image
      │
      ▼
     U-Net
      │
      ▼
Bottleneck Feature Map
      │
      ▼
Class Mask
      │
      ▼
Resize Mask to Feature Resolution
      │
      ▼
Masked Feature Selection
      │
      ▼
Global Average Pooling
      │
      ▼
Class-specific Feature Vector
```

The vector is extracted from the **U-Net bottleneck**, rather than from the final segmentation logits.

With the default architecture, the bottleneck contains 1024 channels, giving a:

```text
1024-dimensional feature vector
```

for each class/document combination where the class is present.

## Ground-Truth vs Predicted Features

Feature extraction is performed using two types of masks:

### Ground-truth representation

```text
Image
  ↓
U-Net
  ↓
Bottleneck
  ↓
Ground-truth class mask
  ↓
Masked average pooling
  ↓
Feature vector
```

### Predicted representation

```text
Image
  ↓
U-Net
  ↓
Bottleneck
  ↓
Predicted class mask
  ↓
Masked average pooling
  ↓
Feature vector
```

This makes it possible to compare how class representations differ when using perfect annotations versus the model's own segmentation.

## Feature Outputs

Feature vectors are stored in:

```text
results/feature_vectors/
```

Example:

```text
image001__class_1_TextRegion.npy
image001__class_3_TableRegion.npy
image001__class_4_ImageRegion.npy
```

Metadata is stored in:

```text
manifest_ground_truth.csv
manifest_predicted.csv
```

Each manifest records:

- image ID
- class ID
- class name
- whether the class is present
- vector filename

## Class Prototypes

For each class, the project computes a prototype:

```text
Class Prototype =
mean(all available document-level feature vectors for that class)
```

Example:

```text
prototype_1_TextRegion.npy
prototype_3_TableRegion.npy
prototype_4_ImageRegion.npy
```

These prototypes can later be used for:

- feature similarity
- clustering
- retrieval
- downstream classification
- representation analysis

## Feature Visualization

Run:

```bash
python visualize.py
```

The current implementation produces a PCA representation:

```text
results/plots/features_pca.png
```

The extracted vectors can subsequently be analyzed using:

- PCA
- t-SNE
- UMAP
- cosine similarity
- nearest-neighbor analysis
- clustering

## Complete Workflow

```text
                    PRImA DATASET
                         │
              ┌──────────┴──────────┐
              │                     │
         Document Images        PAGE XML
              │                     │
              └──────────┬──────────┘
                         ▼
                 XML PARSING
                         │
                         ▼
                PIXEL-WISE MASKS
                         │
                         ▼
               DATA PREPROCESSING
                         │
                         ▼
                 TRAIN / VAL / TEST
                         │
                         ▼
                    U-NET
                         │
              ┌──────────┴──────────┐
              │                     │
         Segmentation          Bottleneck
            Output               Features
              │                     │
              ▼                     ▼
       IoU / Dice / PA       Class Masking
                                    │
                                    ▼
                           Masked Average Pooling
                                    │
                                    ▼
                         Class Feature Vectors
                                    │
                                    ▼
                           Class Prototypes
                                    │
                                    ▼
                       PCA / t-SNE / UMAP / Similarity / Clustering
```

## Expected Output

After running the complete pipeline:

```text
results/
├── checkpoints/
│   └── best.pt
│
├── predictions/
│   └── *.npy
│
├── feature_vectors/
│   ├── manifest_ground_truth.csv
│   ├── manifest_predicted.csv
│   ├── prototype_*.npy
│   └── *.npy
│
├── plots/
│   └── features_pca.png
│
├── history.csv
└── test_metrics.csv
```



## Reproducibility

The default random seed is:

```python
SEED = 42
```

