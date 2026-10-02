

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


def highlightMax(dataframe):
    """
    Highlight the highest numeric value in each row of a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame/table whose row-wise maximum values should be highlighted.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame with the highest value in each row highlighted.

    Notes
    -----
    - Non-numeric values are ignored.
    - Zero values are not highlighted.
    - If a row contains no valid non-zero numeric values, nothing is highlighted.
    - The original DataFrame is not modified.
    """

    # Convert values to numeric for comparison
    numeric_matrix = dataframe.apply(
        pd.to_numeric,
        errors="coerce"
    )

    def highlight_row(row):
        """
        Generate styles for one row.
        """

        values = pd.to_numeric(
            row,
            errors="coerce"
        )

        # Ignore NaN values
        valid_values = values.dropna()

        # If there are no numeric values, highlight nothing
        if valid_values.empty:
            return [""] * len(row)

        # Ignore zero when finding the maximum
        non_zero_values = valid_values[valid_values != 0]

        # If the row contains only zero/NaN values
        if non_zero_values.empty:
            return [""] * len(row)

        # Find the maximum non-zero value
        max_value = non_zero_values.max()

        # Highlight every occurrence of the maximum
        return [
            (
                "background-color: white; "
                "color: black; "
                "font-weight: bold;"
            )
            if pd.notna(value)
            and value == max_value
            else ""
            for value in values
        ]

    return dataframe.style.apply(
        highlight_row,
        axis=1
    )


def highest_similarity(similarity_df):
    """
    Highlight the highest class-to-class cosine similarity
    in each row of the lower-triangular similarity matrix.

    For each row:
    - Only lower-triangular values are considered.
    - The highest value in that row is highlighted.
    - Highlighting is performed using the highlightMax() function.
    - The original similarity_df is not modified.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Original lower-triangular similarity matrix.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled similarity matrix.
    """

    # Create a copy so the original similarity_df is not modified
    lower_triangle_df = similarity_df.copy()

    # Convert numeric cells to a NumPy array
    numeric_matrix = lower_triangle_df.apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy()

    # Create a mask for the lower triangle, excluding diagonal
    lower_triangle_mask = np.tril(
        np.ones_like(
            numeric_matrix,
            dtype=bool
        ),
        k=-1
    )

    # Empty all cells that are NOT in the lower triangle
    # Only for the temporary DataFrame used for highlighting
    for row_index in range(len(lower_triangle_df)):
        for col_index in range(len(lower_triangle_df.columns)):

            if not lower_triangle_mask[row_index, col_index] :
                lower_triangle_df.iloc[
                    row_index,
                    col_index
                ] = ""

    # Highlight the highest value in each row
    return highlightMax(lower_triangle_df)


def max_threshold(similarity_df, x):
    """
    Convert values below threshold x to zero and highlight
    the highest value in each row.

    The original similarity_df is not modified.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Original similarity matrix.

    x : float
        Threshold value. Values below x are converted to zero.

    Returns
    -------
    pandas.io.formats.style.Styler
        Thresholded and highlighted similarity matrix.
    """

    # Create a copy so the original similarity_df is not modified
    thresholded_df = similarity_df.copy()

    # Convert values to numeric for threshold comparison
    numeric_matrix = thresholded_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Convert values below the threshold to zero
    thresholded_df = thresholded_df.mask(
        numeric_matrix < x,
        0
    )

    return thresholded_df