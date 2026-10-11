from imports import *
import random

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import torch


def visualize_class_samples(
    train_dataset,
    test_dataset,
    model,
    device,
    class_to_id,
    class_name_or_id,
    split="test",
    num_samples=1,
    sample_indices=None,
    random_samples=True,
):
    """
    Visualize input images, ground-truth masks, and predicted masks
    for a specific segmentation class.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.

    test_dataset : Dataset
        Test dataset.

    model : torch.nn.Module
        Trained segmentation model.

    device : torch.device
        Device used for inference.

    class_to_id : dict
        Mapping from class names to integer class IDs.
        Example: {"background": 0, "text": 1, "image": 2, ...}

    class_name_or_id : str or int
        Class to visualize. Can be either:
        - a class name, e.g. "noise"
        - a class ID, e.g. 7

    split : {"train", "test"}, default="test"
        Dataset split from which samples are selected.

    num_samples : int, default=1
        Number of samples to visualize.

    sample_indices : list[int] or None, default=None
        Explicit dataset indices to visualize. If provided, these indices
        take priority over random sampling.

    random_samples : bool, default=True
        If True, randomly select samples containing the requested class.
        If False, select the first matching samples.

    Returns
    -------
    selected_indices : list[int]
        Dataset indices of the displayed samples.

    Notes
    -----
    The ground-truth and prediction panels show binary masks for the
    requested class only:
        1 = selected class
        0 = all other classes

    Visualization:
        - The selected class is displayed in pink.
        - Ground truth and prediction masks are overlaid on
          the original image for easier visual comparison.
    """

    # ------------------------------------------------------------------
    # Select dataset
    # ------------------------------------------------------------------
    if split == "train":
        dataset = train_dataset
    elif split == "test":
        dataset = test_dataset
    else:
        raise ValueError("split must be either 'train' or 'test'.")

    if len(dataset) == 0:
        raise ValueError(f"The {split} dataset is empty.")

    # ------------------------------------------------------------------
    # Resolve class name / class ID
    # ------------------------------------------------------------------
    if isinstance(class_name_or_id, str):
        # Try an exact match first.
        if class_name_or_id in class_to_id:
            class_name = class_name_or_id
            class_id = class_to_id[class_name]
        else:
            # Allow case-insensitive class names.
            normalized_name = class_name_or_id.strip().lower()

            matches = {
                name: class_id
                for name, class_id in class_to_id.items()
                if str(name).strip().lower() == normalized_name
            }

            if not matches:
                available_classes = ", ".join(
                    f"{name} ({class_id})"
                    for name, class_id in class_to_id.items()
                )
                raise ValueError(
                    f"Unknown class name: {class_name_or_id!r}. "
                    f"Available classes: {available_classes}"
                )

            class_name, class_id = next(iter(matches.items()))

    elif isinstance(class_name_or_id, (int, np.integer)):
        class_id = int(class_name_or_id)

        reverse_mapping = {
            int(class_id_value): class_name
            for class_name, class_id_value in class_to_id.items()
        }

        if class_id not in reverse_mapping:
            available_ids = sorted(reverse_mapping)
            raise ValueError(
                f"Unknown class ID: {class_id}. "
                f"Available class IDs: {available_ids}"
            )

        class_name = reverse_mapping[class_id]

    else:
        raise TypeError(
            "class_name_or_id must be either a class name (str) "
            "or a class ID (int)."
        )

    # ------------------------------------------------------------------
    # Validate number of samples
    # ------------------------------------------------------------------
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    # ------------------------------------------------------------------
    # Find samples containing the requested class
    # ------------------------------------------------------------------
    matching_indices = []

    for index in range(len(dataset)):
        _, mask = dataset[index]

        mask = mask.cpu().numpy()

        if np.any(mask == class_id):
            matching_indices.append(index)

    if not matching_indices:
        raise ValueError(
            f"No samples in the {split} dataset contain "
            f"class '{class_name}' (ID={class_id})."
        )

    # ------------------------------------------------------------------
    # Select samples
    # ------------------------------------------------------------------
    if sample_indices is not None:
        invalid_indices = [
            index
            for index in sample_indices
            if index not in matching_indices
        ]

        if invalid_indices:
            raise ValueError(
                f"These sample indices do not contain class "
                f"'{class_name}' (ID={class_id}): {invalid_indices}"
            )

        selected_indices = list(sample_indices)[:num_samples]

    else:
        available_indices = matching_indices.copy()

        if random_samples:
            selected_indices = random.sample(
                available_indices,
                min(num_samples, len(available_indices)),
            )
        else:
            selected_indices = available_indices[:num_samples]

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------
    model.eval()

    samples = []

    with torch.inference_mode():
        for sample_index in selected_indices:
            image_tensor, ground_truth_mask = dataset[sample_index]

            # Convert image to H x W x C for matplotlib.
            image = image_tensor.detach().cpu()

            if image.ndim != 3:
                raise ValueError(
                    f"Expected image tensor with 3 dimensions (C, H, W), "
                    f"but received shape {tuple(image.shape)}."
                )

            image = image.permute(1, 2, 0).numpy()

            # If the image was normalized, matplotlib may otherwise show
            # invalid values. Clip only for visualization.
            image = np.clip(image, 0, 1)

            ground_truth_mask = (
                ground_truth_mask.detach().cpu().numpy()
            )

            # Add batch dimension and move to inference device.
            logits = model(
                image_tensor.unsqueeze(0).to(device)
            )

            predicted_mask = (
                logits.argmax(dim=1)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            if predicted_mask.shape != ground_truth_mask.shape:
                raise ValueError(
                    "Predicted mask and ground-truth mask have "
                    f"different shapes for sample {sample_index}: "
                    f"{predicted_mask.shape} vs "
                    f"{ground_truth_mask.shape}."
                )

            # Binary masks for the requested class.
            ground_truth_class_mask = (
                ground_truth_mask == class_id
            ).astype(np.uint8)

            predicted_class_mask = (
                predicted_mask == class_id
            ).astype(np.uint8)

            samples.append(
                {
                    "index": sample_index,
                    "image": image,
                    "ground_truth": ground_truth_class_mask,
                    "prediction": predicted_class_mask,
                }
            )

    # ------------------------------------------------------------------
    # Plot samples
    # ------------------------------------------------------------------
    num_columns = 3
    figure_height = 5 * len(samples)

    fig, axes = plt.subplots(
        len(samples),
        num_columns,
        figsize=(16, figure_height),
        squeeze=False,
    )

    for row, sample in enumerate(samples):
        sample_index = sample["index"]

        axes[row, 0].imshow(sample["image"])
        axes[row, 0].set_title(
            f"Input Image\nSample {sample_index}"
        )

        # Pink class masks overlaid on the original image.
        # Pixels belonging to the selected class are highlighted in pink.
        pink = np.zeros((*sample["ground_truth"].shape, 4), dtype=np.float32)
        pink[..., 0] = 1.0       # Red
        pink[..., 1] = 0.08      # Green
        pink[..., 2] = 0.45      # Blue
        pink[..., 3] = 0.45      # Transparency

        ground_truth_overlay = pink.copy()
        ground_truth_overlay[..., 3] *= sample["ground_truth"]

        prediction_overlay = pink.copy()
        prediction_overlay[..., 3] *= sample["prediction"]

        axes[row, 1].imshow(sample["image"])
        axes[row, 1].imshow(ground_truth_overlay)
        axes[row, 1].set_title(
            f"Ground Truth Overlay: {class_name}\n"
            f"Class ID = {class_id}"
        )

        axes[row, 2].imshow(sample["image"])
        axes[row, 2].imshow(prediction_overlay)
        axes[row, 2].set_title(
            f"Prediction Overlay: {class_name}\n"
            f"Class ID = {class_id}"
        )

        for column in range(num_columns):
            axes[row, column].axis("off")

    fig.suptitle(
        f"Segmentation Samples — {class_name} (ID={class_id})",
        fontsize=16,
        y=1.01,
    )

    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"Class: {class_name}")
    print(f"Class ID: {class_id}")
    print(f"Split: {split}")
    print(f"Samples displayed: {len(selected_indices)}")
    print(f"Selected indices: {selected_indices}")

    for sample in samples:
        gt_pixels = np.count_nonzero(sample["ground_truth"])
        pred_pixels = np.count_nonzero(sample["prediction"])
        total_pixels = sample["ground_truth"].size

        print(
            f"\nSample {sample['index']}:"
            f"\n  Ground truth pixels: {gt_pixels:,} "
            f"({gt_pixels / total_pixels:.2%})"
            f"\n  Predicted pixels:    {pred_pixels:,} "
            f"({pred_pixels / total_pixels:.2%})"
        )

    return selected_indices
