import os
from io import BytesIO
from pathlib import Path

import dropbox
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.reread_preprocessing_utils import (
    add_experiment_IA_numbering,
    add_paragraph_numbering,
    add_first_read_out_of_2_column,
    exclude_IAs,
    remove_nan_cols_and_concat,
    adding_same_diff_critical_span_vars,
)

tqdm.pandas()
# ignore pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'
# set seeds
np.random.seed(42)

root_path = os.getcwd()
data_path = os.path.dirname(root_path) + "/data"


def upload(fig, project, path):
    format = path.split(".")[-1]
    img = BytesIO()
    fig_svg = fig.to_image(format=format)
    img.write(fig_svg)

    token = Path("../dropbox_access_token.txt").read_text()
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=img.getvalue(),
        path=f"/Apps/Overleaf/{project}/{path}",
        mode=dropbox.files.WriteMode.overwrite,
    )


def filter_valid_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where article_ind and IA_ID are greater than 0.

    Parameters:
        df: Input DataFrame.

    Returns:
        Filtered DataFrame.
    """
    return df.query("article_ind > 0 and IA_ID > 0")


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by subject_id, article_ind, unique_paragraph_id, and IA_ID.

    Parameters:
        df: Input DataFrame.

    Returns:
        Sorted DataFrame.
    """
    return df.sort_values(
        by=["subject_id", "article_ind", "unique_paragraph_id", "IA_ID"]
    )


def exclude_and_filter_IAs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude interest areas according to common practice
    and remove subject_id/unique_paragraph_id groups that contain any row
    with Length equal to 0.

    Parameters:
        df: Input DataFrame.

    Returns:
        DataFrame after exclusion and filtering.
    """
    print("Exclude IAs")
    df = exclude_IAs(df)

    # Remove groups with any Length equal to 0.
    df = df.groupby(["subject_id", "unique_paragraph_id"]).filter(
        lambda x: (x["Length"] != 0).all()
    )
    return df


def split_data_by_reread(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the DataFrame based on the 'reread' flag.

    Parameters:
        df: Input DataFrame.

    Returns:
        A tuple of DataFrames: (no_reread_data, reread_data).
    """
    no_reread = df[df["reread"] == 0]
    reread = df[df["reread"] == 1]
    return no_reread, reread


def concatenate_reread_data(
    reread_df: pd.DataFrame, no_reread_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove columns containing 'prev' from one of the DataFrames and
    concatenate the reread and no-reread DataFrames using an external helper.

    Parameters:
        reread_df: DataFrame with reread == 1.
        no_reread_df: DataFrame with reread == 0.

    Returns:
        Concatenated DataFrame with removed nan columns.
    """
    # Identify columns that contain 'prev'
    cols_to_ignore = [col for col in no_reread_df.columns if "prev" in col]
    return remove_nan_cols_and_concat(
        reread_df, no_reread_df, cols_to_ignore=cols_to_ignore
    )


def add_paragraph_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'paragraph_len' which represents the number of interest areas
    (i.e. rows) for each unique paragraph.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with the joined paragraph length information.
    """
    print("Adding paragraph_length column...")
    # Get unique (unique_paragraph_id, IA_ID) pairs
    paragraph_only_IA = df[["unique_paragraph_id", "IA_ID"]].drop_duplicates()
    # Count interest areas per paragraph
    paragraph_len_df = (
        paragraph_only_IA.groupby("unique_paragraph_id")
        .size()
        .rename("paragraph_len")
        .reset_index()
    )
    # Join the paragraph length to the main DataFrame
    df = df.join(
        paragraph_len_df.set_index("unique_paragraph_id"), on="unique_paragraph_id"
    )
    return df


def split_unique_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the 'unique_paragraph_id' column into several distinct columns
    (batch, article_id, level, paragraph_id) and extracts 'list' from subject_id.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with new columns added.
    """
    print("Splitting unique_paragraph_id to original columns")
    df["batch"] = df["unique_paragraph_id"].apply(lambda x: x.split("_")[0]).astype(int)
    df["article_id"] = (
        df["unique_paragraph_id"].apply(lambda x: x.split("_")[1]).astype(int)
    )
    df["level"] = df["unique_paragraph_id"].apply(lambda x: x.split("_")[2])
    df["paragraph_id"] = (
        df["unique_paragraph_id"].apply(lambda x: x.split("_")[3]).astype(int)
    )
    df["list"] = df["subject_id"].apply(lambda x: x.split("_")[0][1:]).astype(int)
    return df


def add_has_preview_reread_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'has_preview_reread' by concatenating 'has_preview' and 'reread'.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with the new column.
    """
    df["has_preview_reread"] = (
        df["has_preview"].astype(str) + ", " + df["reread"].astype(str)
    )
    return df


def apply_first_read_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the indicator for the 'first read out of 2' using a pre-defined helper.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with the indicator column added.
    """
    print("Adding add_first_read_out_of_2_column indicator column...")
    return add_first_read_out_of_2_column(df)


def apply_experiment_IA_numbering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the experiment IA numbering indicator column into the DataFrame.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with the experiment IA numbering column merged.
    """
    print("Adding experiment_IA_numbering indicator column...")
    experiment_df = (
        df.groupby("subject_id")
        .apply(add_experiment_IA_numbering)
        .reset_index()
        .drop("level_1", axis=1)
    )
    df = df.merge(experiment_df, on=["subject_id", "article_ind", "IA_ID"], how="left")
    return df


def apply_paragraph_numbering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds paragraph numbering using a pre-defined helper.

    Parameters:
        df: DataFrame to update.

    Returns:
        DataFrame with paragraph numbering added.
    """
    print("Adding paragraph_numbering column...")
    return add_paragraph_numbering(df)


def add_diffs_from_first_reading(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the differences between reread trials for specific columns compared to the first reading.
    Computes the diff columns for article_ind, paragraph_numbering, and experiment_IA_numbering,
    then creates corresponding columns for the first reading.

    Parameters:
        df: DataFrame to update.

    Returns:
        Updated DataFrame with difference and first reading columns added.
    """
    print("Adding article_ind_diff_from_first_reading column for articles 11 and 12...")
    diff_cols = [
        "article_ind",
        "paragraph_numbering",
        "experiment_IA_numbering",
        "reread",
    ]

    # Prepare pivot table:
    pivotdf = (
        df[
            [
                "subject_id",
                "unique_paragraph_id",
                "IA_ID",
                "article_ind",
                "paragraph_numbering",
                "experiment_IA_numbering",
                "reread",
            ]
        ]
        .drop_duplicates()
        .pivot(
            index=["subject_id", "unique_paragraph_id", "IA_ID"],
            columns="reread",
            values=diff_cols,
        )
        .dropna()
    )

    # Compute difference between reread (1) and first reading (0)
    for col in diff_cols:
        pivotdf[col] = pivotdf[col][1] - pivotdf[col][0]
        if col != "reread":
            pivotdf.rename(
                columns={col: col + "_diff_from_first_reading"}, inplace=True
            )

    # Simplify the pivot column structure and merge back with original DataFrame:
    pivotdf.columns = pivotdf.columns.droplevel(1)
    pivotdf = pivotdf.loc[:, ~pivotdf.columns.duplicated()].reset_index()

    df = pd.merge(
        df,
        pivotdf,
        on=["subject_id", "unique_paragraph_id", "IA_ID", "reread"],
        how="left",
    )

    # Create first reading columns by subtracting the difference:
    df["article_ind_of_first_reading"] = (
        df["article_ind"] - df["article_ind_diff_from_first_reading"]
    )
    df["paragraph_numbering_of_first_reading"] = (
        df["paragraph_numbering"] - df["paragraph_numbering_diff_from_first_reading"]
    )
    df["experiment_IA_numbering_of_first_reading"] = (
        df["experiment_IA_numbering"]
        - df["experiment_IA_numbering_diff_from_first_reading"]
    )
    return df


def process_fixation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes various fixation/count columns:
      - Converts '.' to NaN in IA_REGRESSION_OUT_FULL_COUNT and casts to numeric.
      - Creates IA_FIRST_PASS_GAZE_DURATION from IA_FIRST_RUN_DWELL_TIME only when IA_FIRST_FIX_PROGRESSIVE == 1.
      - Creates an indicator column IA_ZERO_TF for rows where IA_DWELL_TIME == 0.

    Parameters:
        df: DataFrame to update.

    Returns:
        Updated DataFrame with processed fixation columns.
    """
    # Replace '.' with nan and convert the column to numeric
    df["IA_REGRESSION_OUT_FULL_COUNT"] = df["IA_REGRESSION_OUT_FULL_COUNT"].replace(
        ".", np.nan
    )
    df["IA_REGRESSION_OUT_FULL_COUNT"] = pd.to_numeric(
        df["IA_REGRESSION_OUT_FULL_COUNT"]
    )

    # Create IA_FIRST_PASS_GAZE_DURATION conditionally
    df["IA_FIRST_PASS_GAZE_DURATION"] = df["IA_FIRST_RUN_DWELL_TIME"]
    df.loc[df["IA_FIRST_FIX_PROGRESSIVE"] != 1, "IA_FIRST_PASS_GAZE_DURATION"] = np.nan

    # Indicator for zero dwell time
    df["IA_ZERO_TF"] = df["IA_DWELL_TIME"] == 0
    return df


def apply_same_diff_critical_span_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds variables indicating whether rereading trials have the same or a different
    critical span as in the first reading, using a pre-defined helper.

    Parameters:
        df: DataFrame to update.

    Returns:
        Updated DataFrame with same/different critical span variables.
    """
    print(
        "Adding variables indicating rereading trials with the same / different critical span as in the first reading"
    )
    return adding_same_diff_critical_span_vars(df)


def process_df_for_reread_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame for reread analysis by applying a series of cleaning,
    transformation, and feature extraction steps. This unified function combines:
      - Data filtering and sorting.
      - Exclusion of certain interest areas and filtering based on Length.
      - Splitting data by reread status and concatenating.
      - Adding paragraph length, splitting unique IDs to original columns,
        and adding a combined 'has_preview_reread' column.
      - Adding first reading indicators, experiment IA numbering, paragraph numbering.
      - Computing differences from the first reading.
      - Processing fixation and related count columns.
      - Adding variables for critical span differences.

    Parameters:
        data: Raw input DataFrame.

    Returns:
        Processed DataFrame ready for further analysis.
    """
    # Step 1: Filter out invalid articles and sort the data.
    data = filter_valid_articles(data)
    data = sort_data(data)

    # Step 2: Exclude unwanted IAs and filter out groups with Length == 0.
    data = exclude_and_filter_IAs(data)

    # Step 3: Split into no-reread and reread groups, then concatenate.
    no_reread, reread = split_data_by_reread(data)
    data = concatenate_reread_data(reread, no_reread)

    # Step 4: Add paragraph length.
    data = add_paragraph_length(data)

    # Step 5: Split composite unique_paragraph_id and subject_id columns.
    data = split_unique_id_columns(data)

    # Step 6: Create a combined has_preview_reread column.
    data = add_has_preview_reread_column(data)

    # Step 7: Add first reading indicator column.
    data = apply_first_read_indicator(data)

    # Step 8: Add experiment IA numbering.
    data = apply_experiment_IA_numbering(data)

    # Step 9: Add paragraph numbering.
    data = apply_paragraph_numbering(data)

    # Step 10: Compute differences from the first reading.
    data = add_diffs_from_first_reading(data)

    # Step 11: Process fixation-related columns.
    data = process_fixation_columns(data)

    # Step 12: Add same/different critical span variables.
    data = apply_same_diff_critical_span_vars(data)

    return data


def main():
    print("Loading data and removing NaN columns...")
    data = pd.read_csv("data/interim/ia_data_enriched_yoav_cogsci_all_participants.csv")

    et_data_enriched = process_df_for_reread_analysis(data)

    saving_path = (
        data_path + "/interim/et_data_for_reread_analysis_all_participants.csv"
    )
    print("Saving data to...")
    print(saving_path)
    et_data_enriched.to_csv(saving_path, index=False)


if __name__ == "__main__":
    main()
