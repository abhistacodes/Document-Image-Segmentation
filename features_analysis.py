from imports import *
import numpy as np
import pandas as pd


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

            if not lower_triangle_mask[row_index, col_index] and row_index != col_index:
                lower_triangle_df.iloc[
                    row_index,
                    col_index
                ] = 0.0

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


def print_highest_values(dataframe):
    """
    Print the [row, column] labels of the highest non-zero numeric
    value(s) in each row of a DataFrame.

    This follows the same row-wise rules as highlightMax():
    - Non-numeric and NaN values are ignored.
    - Zero values are ignored.
    - If multiple cells tie for the row maximum, all are printed.
    - Rows with no valid non-zero numeric values are skipped.
    - The input DataFrame is not modified.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame whose row-wise maximum cell labels should be printed.

    Returns
    -------
    list[tuple]
        List of (row_label, column_label) pairs for the maximum cell(s).
    """
    highlighted_cells = []

    for row_label, row in dataframe.iterrows():
        values = pd.to_numeric(row, errors="coerce")
        valid_values = values.dropna()
        non_zero_values = valid_values[valid_values != 0]

        if non_zero_values.empty:
            continue

        max_value = non_zero_values.max()

        for column_label, value in values.items():
            if pd.notna(value) and value == max_value:
                cell = (row_label, column_label)
                highlighted_cells.append(cell)
                print(f"[{row_label}, {column_label}]")

    return highlighted_cells


def class_wise_highest_similarity(similarity_df):
    """
    Find each class's highest similarity with any other class by checking
    both its row and its column.

    This supports lower-triangular similarity DataFrames where a pair's
    similarity may appear in either the class's row or its column.

    Parameters
    ----------
    similarity_df : pandas.DataFrame
        Similarity matrix with class names as both index and columns.

    Returns
    -------
    dict
        Maps each class label to a list of (class_label, similarity_value)
        pairs tied for that class's highest non-zero similarity.
    """
    results = {}

    for class_name in similarity_df.index:
        candidates = []

        # Values in this class's row: compare with classes in the columns.
        if class_name in similarity_df.index:
            row = similarity_df.loc[class_name]
            for other_class, value in row.items():
                if other_class == class_name:
                    continue
                numeric_value = pd.to_numeric(
                    pd.Series([value]), errors="coerce"
                ).iloc[0]
                if pd.notna(numeric_value) and numeric_value != 0:
                    candidates.append((other_class, float(numeric_value)))

        # Values in this class's column: compare with classes in the rows.
        if class_name in similarity_df.columns:
            column = similarity_df[class_name]
            for other_class, value in column.items():
                if other_class == class_name:
                    continue
                numeric_value = pd.to_numeric(
                    pd.Series([value]), errors="coerce"
                ).iloc[0]
                if pd.notna(numeric_value) and numeric_value != 0:
                    candidates.append((other_class, float(numeric_value)))

        if not candidates:
            results[class_name] = []
            print(f"[{class_name}, no non-zero similarity found]")
            continue

        highest_value = max(value for _, value in candidates)

        # Keep all classes tied at the highest value, without duplicate pairs.
        highest_classes = []
        seen = set()
        for other_class, value in candidates:
            if value == highest_value and other_class not in seen:
                highest_classes.append((other_class, value))
                seen.add(other_class)

        results[class_name] = highest_classes
        for other_class, value in highest_classes:
            print(f"[{class_name}, {other_class}] = {value:.6f}")

    return results
