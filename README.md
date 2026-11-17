# Document Image Segmentation and Feature Analysis

## Overview
This project focuses on document image segmentation and class-specific feature analysis using deep learning. It uses the **PRImA Layout Analysis Dataset** to segment document images into their respective layout classes using a U-Net-based architecture and subsequently extract representative deep feature vectors for each class.

## Objectives
- Perform pixel-level segmentation of document images.
- Identify and classify different document layout elements.
- Train a U-Net model for semantic segmentation.
- Extract class-specific deep representative feature vectors.
- Analyze relationships between document classes using cosine similarity.
- Identify highly similar classes and investigate their feature-space relationships through hierarchical clustering and dendrogram analysis.

## Methodology

The project follows the pipeline below:

1. **Dataset Preparation:** Load, preprocess, and validate document images and their corresponding ground-truth annotations from the PRImA dataset.
2. **Semantic Segmentation:** Train a U-Net architecture to generate pixel-level segmentation masks for document layout classes.
3. **Feature Extraction:** Extract deep feature representations from the trained model using class-specific segmentation masks.
4. **Prototype Generation:** Compute representative feature vectors (class prototypes) for each layout class.
5. **Similarity Analysis:** Calculate pairwise cosine similarities between class prototypes and generate similarity matrices.
6. **Hierarchical Clustering:** Analyze class relationships, generate candidate dendrogram cuts, and investigate groups of similar document classes.

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset
**PRImA Layout Analysis Dataset**

The dataset contains annotated document images with multiple layout classes. Original annotations are processed to generate segmentation masks suitable for semantic segmentation and subsequent feature analysis.

## Project Structure

```text
project/
├── data/
│   ├── images/
│   └── annotations/
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_dataset_validation.ipynb
│   ├── 04_baseline_model.ipynb
│   └── 05_training.ipynb
├── models/
├── features/
├── visualizations/
├── scripts/
└── README.md
```

*Note: Directory names are illustrative and should be adjusted to match the actual repository structure.*

## Current Progress
- Dataset preparation, preprocessing, and validation.
- U-Net-based semantic segmentation pipeline.
- Extraction of class-specific deep feature representations.
- Generation and storage of class prototypes.
- Cosine similarity analysis between class prototypes.
- Lower-triangular similarity visualization and threshold-based analysis.
- Hierarchical clustering and dendrogram-based exploration of class relationships.

## Future Scope
- Refine segmentation performance.
- Evaluate feature representations across different classes.
- Investigate optimal clustering configurations.
- Validate discovered class relationships using quantitative metrics.

## License
This project is intended for academic and research purposes.