import numpy as np
import pandas as pd


def highlightMax(dataframe):
    """
    Highlight every row-wise maximum in a DataFrame.

    Non-numeric values and zeros are ignored. If a row has no non-zero
    numeric values, no cell in that row is highlighted. Tied maxima are
    all highlighted.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame whose row-wise maximum values should be highlighted.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame with the highest value(s) in each row highlighted.
    """
    numeric_df = dataframe.apply(pd.to_numeric, errors="coerce")

    def style_row(row):
        values = pd.to_numeric(row, errors="coerce")
        candidates = values[values.notna() & values.ne(0)]

        styles = pd.Series("", index=row.index, dtype=object)
        if not candidates.empty:
            maximum = candidates.max()
            styles.loc[values.eq(maximum)] = (
                "background-color: yellow; color: black; font-weight: bold;"
            )
        return styles.tolist()

    return dataframe.style.apply(style_row, axis=1)


def highest_similarity(similarity_df):
    """
    Highlight the highest non-zero similarity value(s) in each row.

    Works with a full similarity matrix or a lower-triangular DataFrame
    whose unavailable cells are blank/NaN. The input DataFrame is not
    modified. Tied maxima are all highlighted.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Full or lower-triangular similarity matrix.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled copy of the input DataFrame.
    """
    dataframe_copy = similarity_df.copy()
    return highlightMax(dataframe_copy)


def max_threshold(similarity_df, x):
    """
    Return a copy of the DataFrame with numeric values below threshold x
    replaced by zero.

    Non-numeric cells and NaN values are preserved. The input DataFrame
    is not modified.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Similarity matrix to threshold.
    x : float
        Threshold; values strictly less than x become zero.

    Returns
    -------
    pandas.DataFrame
        A thresholded copy of the input DataFrame.
    """
    thresholded_df = similarity_df.copy()
    numeric_df = thresholded_df.apply(pd.to_numeric, errors="coerce")
    below_threshold = numeric_df.lt(x)

    # Replace only cells that are numeric and below the threshold.
    return thresholded_df.mask(below_threshold, 0)


def class_wise_highest_similarity(similarity_df):
    """
    Find each class's highest non-zero similarity with any other class,
    considering both its row and its column.

    This supports full or lower-triangular matrices. Diagonal entries,
    zeros, NaNs, and non-numeric values are ignored. Ties are retained,
    and duplicate class pairs are returned only once.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Similarity matrix with class labels in the index and columns.

    Returns
    -------
    dict
        Maps each class label to a list of (other_class, similarity_value)
        pairs tied for that class's highest non-zero similarity.
    """
    results = {}

    for class_name in similarity_df.index:
        # Store the best value per other class to avoid duplicate candidates
        # when the same pair appears in both the row and the column.
        candidates = {}

        if class_name in similarity_df.columns:
            row = similarity_df.loc[class_name]
            for other_class, value in row.items():
                if other_class == class_name:
                    continue
                numeric_value = pd.to_numeric(
                    pd.Series([value]), errors="coerce"
                ).iloc[0]
                if pd.notna(numeric_value) and numeric_value != 0:
                    candidates[other_class] = max(
                        candidates.get(other_class, -np.inf),
                        float(numeric_value),
                    )

        if class_name in similarity_df.columns:
            column = similarity_df[class_name]
            for other_class, value in column.items():
                if other_class == class_name:
                    continue
                numeric_value = pd.to_numeric(
                    pd.Series([value]), errors="coerce"
                ).iloc[0]
                if pd.notna(numeric_value) and numeric_value != 0:
                    candidates[other_class] = max(
                        candidates.get(other_class, -np.inf),
                        float(numeric_value),
                    )

        if not candidates:
            results[class_name] = []
            continue

        highest_value = max(candidates.values())
        results[class_name] = [
            (other_class, value)
            for other_class, value in candidates.items()
            if value == highest_value
        ]

    return results
