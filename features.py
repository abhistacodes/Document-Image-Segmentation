

from imports import *
import numpy as np
import pandas as pd

#MASK GUIDED DEEP FEATURE EXTRACTION

@torch.no_grad()
def extract_class_features(
    model,
    image_tensor,
    segmentation_mask,
    num_classes,
    device
):
    """
    Extract one deep feature vector for each class present
    in a document image.

    Parameters
    ----------
    model:
        Trained U-Net.

    image_tensor:
        Tensor of shape [B, 3, H, W].

    segmentation_mask:
        Tensor of shape [B, H, W].
        Each pixel contains a class ID.

    num_classes:
        Number of segmentation classes.

    device:
        CPU or CUDA.

    Returns
    -------
    class_features:
        Tensor [B, num_classes, 1024].

        If a class is absent from an image, its vector is NaN.

    class_present:
        Tensor [B, num_classes].
        True if the class occurs in the image.
    """

    model.eval()

    image_tensor = image_tensor.to(device)
    segmentation_mask = segmentation_mask.to(device)

    #obtain logits and feature map from the model

    logits, feature_map = model(
        image_tensor,
        return_features=True
    )

    # feature_map:
    # [B, 1024, Hf, Wf]

    B, C, Hf, Wf = feature_map.shape

    #resize segmentation mask to feature-map resolution

    resized_mask = F.interpolate(
        segmentation_mask.unsqueeze(1).float(),
        size=(Hf, Wf),
        mode="nearest"
    ).squeeze(1).long()

    #storage

    class_features = torch.full(
        (B, num_classes, C),
        float("nan"),
        device=device
    )

    class_present = torch.zeros(
        (B, num_classes),
        dtype=torch.bool,
        device=device
    )

    
    #extract features for each class present in the image
    for class_id in range(num_classes):

        # Binary mask for this class
        class_mask = (
            resized_mask == class_id
        )

        # Number of pixels belonging to class
        pixel_count = class_mask.sum(
            dim=(1, 2)
        )

        present = pixel_count > 0

        class_present[:, class_id] = present

        #mask guided global average pooling of the feature map
        mask_float = class_mask.float()

        masked_features = (
            feature_map *
            mask_float.unsqueeze(1)
        )

        feature_sum = masked_features.sum(
            dim=(2, 3)
        )

        denominator = pixel_count.clamp(
            min=1
        ).unsqueeze(1).float()

        pooled_features = (
            feature_sum /
            denominator
        )

        class_features[:, class_id, :] = torch.where(
            present.unsqueeze(1),
            pooled_features,
            torch.full_like(
                pooled_features,
                float("nan")
            )
        )

    return (
        class_features,
        class_present
    )


def highest_similarity(similarity_df):
    """
    Highlight the highest class-to-class cosine similarity
    in each row of the lower-triangular similarity matrix.

    For each row:
    - Only lower-triangular values are considered.
    - The highest value in that row is highlighted.
    - Yellow background, black font, and bold text are used.
    """

    # Convert numeric cells to a NumPy array
    numeric_matrix = similarity_df.apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy()

    # Create a mask for the lower triangle, excluding diagonal
    lower_triangle_mask = np.tril(
        np.ones_like(numeric_matrix, dtype=bool),
        k=-1
    )

    # Ignore values outside the lower triangle
    lower_triangle = np.where(
        lower_triangle_mask,
        numeric_matrix,
        np.nan
    )

    # Find the maximum value in each row
    row_max = np.nanmax(
        lower_triangle,
        axis=1
    )

    # Find positions of row-wise maximum values
    max_positions = np.zeros_like(
        numeric_matrix,
        dtype=bool
    )

    for row_index in range(len(row_max)):
        if not np.isnan(row_max[row_index]):
            max_positions[row_index, :] = (
                lower_triangle[row_index, :] == row_max[row_index]
            )

    # Highlight the maximum value in each row
    def highlight_max(row):
        row_index = similarity_df.index.get_loc(row.name)

        styles = [""] * len(row)

        for col_index in range(len(row)):
            if max_positions[row_index, col_index]:
                styles[col_index] = (
                    "background-color: yellow; "
                    "color: black; "
                    "font-weight: bold;"
                )

        return styles

    
    return similarity_df.style.apply(
        highlight_max,
        axis=1
    )