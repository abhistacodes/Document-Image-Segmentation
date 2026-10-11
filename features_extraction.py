from imports import *
import numpy as np
import pandas as pd

#MASK GUIDED DEEP FEATURE EXTRACTION

@torch.no_grad()
def extract_class_features(
    model,
    image_tensor,
    segmentation_mask,
    class_ids,
    device
):
    """
    Extract feature vectors only for the requested original segmentation IDs.

    class_ids must contain the original mask IDs in the desired output order.
    Returns class_features [B, len(class_ids), C] and class_present
    [B, len(class_ids)]. Missing classes have NaN feature vectors.
    """
    
    model.eval()
    image_tensor = image_tensor.to(device)
    segmentation_mask = segmentation_mask.to(device)
    class_ids = list(class_ids)

    _, feature_map = model(image_tensor, return_features=True)
    B, C, Hf, Wf = feature_map.shape

    resized_mask = F.interpolate(
        segmentation_mask.unsqueeze(1).float(),
        size=(Hf, Wf),
        mode="nearest"
    ).squeeze(1).long()

    class_features = torch.full(
        (B, len(class_ids), C), float("nan"), device=device
    )
    class_present = torch.zeros(
        (B, len(class_ids)), dtype=torch.bool, device=device
    )

    # Output index differs from original class ID: only selected classes
    # receive output slots, while masks are compared against original IDs.
    for output_index, class_id in enumerate(class_ids):
        class_mask = resized_mask == class_id
        pixel_count = class_mask.sum(dim=(1, 2))
        present = pixel_count > 0
        class_present[:, output_index] = present

        masked_features = feature_map * class_mask.float().unsqueeze(1)
        feature_sum = masked_features.sum(dim=(2, 3))
        denominator = pixel_count.clamp(min=1).unsqueeze(1).float()
        pooled_features = feature_sum / denominator

        class_features[:, output_index, :] = torch.where(
            present.unsqueeze(1),
            pooled_features,
            torch.full_like(pooled_features, float("nan"))
        )

    return class_features, class_present
