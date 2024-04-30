import multiprocessing
import os
import re
from io import BytesIO
from pathlib import Path

import dropbox
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

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

    token = Path(
        "../dropbox_access_token.txt"
    ).read_text()
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=img.getvalue(),
        path=f"/Apps/Overleaf/{project}/{path}",
        mode=dropbox.files.WriteMode.overwrite,
    )


def check_same_span_2_trials(df):
    read_is_in_span = (
        df.loc[lambda x: x["reread"] == 0]
        .sort_values(by=["IA_ID"])["relative_to_aspan"]
        .reset_index(drop=True)
    )
    reread_is_in_span = (
        df.loc[lambda x: x["reread"] == 1]
        .sort_values(by=["IA_ID"])["relative_to_aspan"]
        .reset_index(drop=True)
    )
    assert len(read_is_in_span) == len(reread_is_in_span)
    return (read_is_in_span == reread_is_in_span).all()

def exclude_IAs(
    df: pd.DataFrame, remove_start_end_of_line: bool = True
) -> pd.DataFrame:
    et_data_enriched = df.copy()
    # ? Remove first and last words in each paragraph
    # For every unique_paragraph_id, subject_id, reread triplet, find the maximal and minimal IA_IDs
    # and remove the records with the minimal and maximal IA_ID
    min_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .min()
        .reset_index()
    )
    max_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .max()
        .reset_index()
    )

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in min_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        min_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in max_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        max_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # ? Remove words that are not all letters (contains numbers or symbols inclusind punctuation)
    et_data_enriched = et_data_enriched.loc[
        et_data_enriched["IA_LABEL"].apply(lambda x: bool(re.match("^[a-zA-Z ]*$", x)))
    ]

    if remove_start_end_of_line:
        # if 'end of line' column is in the dataframe, remove all rows where 'end of line' == 1
        if "end_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.query("end_of_line == False")

        if "start_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.query("start_of_line == False")

    return et_data_enriched


def add_experiment_IA_numbering(subject_df):
    IA_ID_experiment_numbering = (
        subject_df[["article_ind", "IA_ID"]]
        .drop_duplicates()
        .sort_values(by=["article_ind", "IA_ID"])
    )
    # add numbering column which is according to the row order
    IA_ID_experiment_numbering["experiment_IA_numbering"] = np.arange(
        1, len(IA_ID_experiment_numbering) + 1
    )
    return IA_ID_experiment_numbering.reset_index(drop=True)


def create_subject_article_ind_diff_df(df):
    subjects_article_indices_12_and_first = df.query(
        "first_read_out_of_2 == True | article_ind == 12"
    )[["subject_id", "article_id", "article_ind"]].drop_duplicates()

    # print(
    #     len(
    #         subjects_article_indices_12_and_first.groupby(["subject_id"]).filter(
    #             lambda x: len(x) == 2
    #         )
    #     )
    #     == len(subjects_article_indices_12_and_first)
    # )

    subjects_article_indices_12 = subjects_article_indices_12_and_first.query(
        "article_ind == 12"
    )
    subjects_article_indices_first = subjects_article_indices_12_and_first.query(
        "article_ind != 12"
    )

    subject_article_ind_diff = pd.merge(
        subjects_article_indices_12,
        subjects_article_indices_first,
        on=["subject_id", "article_id"],
        suffixes=("_12", "_first"),
    )
    subject_article_ind_diff["ind_diff"] = (
        subject_article_ind_diff["article_ind_12"]
        - subject_article_ind_diff["article_ind_first"]
    )
    subject_article_ind_diff.drop("article_ind_first", axis=1, inplace=True)
    return subject_article_ind_diff


def remove_nan_cols_and_concat(
    et_data_enriched_reread: pd.DataFrame,
    et_data_enriched_no_reread: pd.DataFrame,
    cols_to_ignore: list = [],
):
    # print the column names with nan values
    nan_cols_no_rerad = et_data_enriched_no_reread.columns[
        et_data_enriched_no_reread.isna().any()
    ].tolist()
    nan_cols_rerad = et_data_enriched_reread.columns[
        et_data_enriched_reread.isna().any()
    ].tolist()

    # create a union of the two lists
    nan_cols = list(set(nan_cols_no_rerad + nan_cols_rerad))

    # from et_data_enriched_no_reread drop all columns that are in nan_cols
    nan_cols_that_are_in_et_data_enriched_reread = [
        col
        for col in nan_cols
        if col in et_data_enriched_reread.columns and col not in cols_to_ignore
    ]
    nan_cols_that_are_in_et_data_enriched_no_reread = [
        col
        for col in nan_cols
        if col in et_data_enriched_no_reread.columns and col not in cols_to_ignore
    ]
    et_data_enriched_no_reread = et_data_enriched_no_reread.drop(
        nan_cols_that_are_in_et_data_enriched_no_reread, axis=1
    )
    # from et_data_enriched_reread drop all columns that are in nan_cols
    et_data_enriched_reread = et_data_enriched_reread.drop(
        nan_cols_that_are_in_et_data_enriched_reread, axis=1
    )
    et_data_enriched = pd.concat([et_data_enriched_no_reread, et_data_enriched_reread])
    return et_data_enriched


def add_first_read_out_of_2_column(et_data_enriched: pd.DataFrame):
    """create a new column 'first_read_out_of_2' in et_data_enriched which is
    1 if the columns ['subject_id', 'unique_paragraph_id', 'reread']
    are in trials_first_read_out_of_2 and 0 otherwise. Dont use itterrows
    """
    trials_first_read_out_of_2 = (
        et_data_enriched[["subject_id", "unique_paragraph_id", "reread"]]
        .drop_duplicates()  # Get all unique trials
        .groupby(["subject_id", "unique_paragraph_id"])
        .filter(lambda x: len(x) > 1)
        .loc[lambda x: x["reread"] == 0]  # Get only the first read
        .astype({"reread": int})
    ).set_index(["subject_id", "unique_paragraph_id", "reread"])
    et_data_enriched["first_read_out_of_2"] = False
    # use pd.merge somehow
    et_data_enriched = et_data_enriched.set_index(
        ["subject_id", "unique_paragraph_id", "reread"]
    )
    et_data_enriched.loc[
        et_data_enriched.index.isin(trials_first_read_out_of_2.index),
        "first_read_out_of_2",
    ] = True
    et_data_enriched = et_data_enriched.reset_index()

    return et_data_enriched


def sort_and_add_numbering(
    df, column_names, ascending=True, numbering_name="numbering"
):
    df = df.sort_values(column_names, ascending=ascending)
    df = df.reset_index(drop=True)
    df[numbering_name] = df.index + 1
    return df


def add_paragraph_numbering(et_data_enriched: pd.DataFrame):
    paragraph_numbering = (
        et_data_enriched[
            ["subject_id", "article_ind", "paragraph_id", "unique_paragraph_id"]
        ]
        .drop_duplicates()
        .groupby(["subject_id"])
        .apply(
            lambda x: sort_and_add_numbering(
                x,
                ["article_ind", "paragraph_id"],
                ascending=True,
                numbering_name="paragraph_numbering",
            )
        )
        .reset_index(drop=True)
    )

    et_data_enriched = et_data_enriched.merge(
        paragraph_numbering,
        on=["subject_id", "article_ind", "paragraph_id", "unique_paragraph_id"],
        how="left",
    )

    return et_data_enriched


def applyParallel(
    dfGrouped: pd.core.groupby.generic.DataFrameGroupBy,
    func: callable,
    ):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in dfGrouped
    )
    return pd.concat(retLst)


def compute_difference_reread(group):
    reread_1 = group[group["reread"] == 1].set_index(
        ["subject_id", "unique_paragraph_id", "IA_ID"]
    )
    reread_0 = group[group["reread"] == 0].set_index(
        ["subject_id", "unique_paragraph_id", "IA_ID"]
    )
    diff_df = reread_1 - reread_0
    # add to all columns the suffix _diff
    diff_df.columns = [col + "_diff_from_first_reading" for col in diff_df.columns]
    # drop the index and remove it's columns (including the column that is all zeros)
    diff_df = diff_df.reset_index()

    return diff_df

def adding_same_diff_critical_span_vars(et_df: pd.DataFrame):
    same_span_in_12_reread = (
        et_df.query(
            "(first_read_out_of_2 == 1 and article_ind < 10) or article_ind == 12"
        )
        .groupby(["subject_id", "unique_paragraph_id"])
        .apply(lambda x: check_same_span_2_trials(x))
    )
    same_span_in_11_reread = (
        et_df.query("article_ind == 10 or article_ind == 11")
        .groupby(["subject_id", "unique_paragraph_id"])
        .apply(lambda x: check_same_span_2_trials(x))
    )

    same_span_in_12_reread = same_span_in_12_reread.reset_index().rename(
        columns={0: "same_span_in_FR"}
    )
    same_span_in_11_reread = same_span_in_11_reread.reset_index().rename(
        columns={0: "same_span_in_FR"}
    )

    # concatenate them to one df
    same_span_in_12_reread["article_ind"] = 12
    same_span_in_11_reread["article_ind"] = 11

    same_span_in_FR = pd.concat(
        [same_span_in_12_reread, same_span_in_11_reread], axis=0
    )

    # "same_question_as_FR" will be '.' for reread == 0 and same_span_in_12_reread for reread == 1 (joined on subject_id, unique_paragraph_id). Replace nans with '.'
    et_df = et_df.merge(
        same_span_in_FR,
        on=["subject_id", "unique_paragraph_id", "article_ind"],
        how="left",
        suffixes=("", "_y"),
    )
    # replace nans with '.'
    et_df["same_span_in_FR"] = et_df["same_span_in_FR"].fillna(
        "."
    )

    # put et_data_enriched['same_span_in_FR'] == '.' were et_data_enriched['has_preview'] == Gathering
    et_df.loc[
        et_df["has_preview"] == "Gathering", "same_span_in_FR"
    ] = "."

    same_span_in_FR_also_gathering_df = same_span_in_FR.copy()
    # rename the same_span_in_FR column to same_span_in_FR_also_gathering
    same_span_in_FR_also_gathering_df = same_span_in_FR_also_gathering_df.rename(
        columns={"same_span_in_FR": "same_span_in_FR_also_gathering"}
    )
    et_df = et_df.merge(
        same_span_in_FR_also_gathering_df,
        on=["subject_id", "unique_paragraph_id", "article_ind"],
        how="left",
        suffixes=("", "_y"),
    )
    et_df.loc[
        et_df["article_ind"] < 11, "same_span_in_FR_also_gathering"
    ] = "."
    
    return et_df

def process_df_for_reread_analysis(data: pd.DataFrame):
    data = data.query("article_ind > 0 and IA_ID > 0")

    data = data.sort_values(
        by=["subject_id", "article_ind", "unique_paragraph_id", "IA_ID"]
    )
    
    # exclude interest areas following common practice:
    print("Exclude IAs")
    data = exclude_IAs(data)
    # remove all subject_id, paragraph_id pairs which have a record with Length == 0
    data = data.groupby(
        ["subject_id", "unique_paragraph_id"]
    ).filter(lambda x: (x["Length"] != 0).all())

    
    et_data_enriched_no_reread = data[data["reread"] == 0]
    et_data_enriched_reread = data[data["reread"] == 1]

    # cols_to_ignore is all columns that contain the word 'prev'
    cols_to_ignore = [
        col for col in et_data_enriched_no_reread.columns if "prev" in col
    ]

    et_data_enriched = remove_nan_cols_and_concat(
        et_data_enriched_reread,
        et_data_enriched_no_reread,
        cols_to_ignore=cols_to_ignore,
    )
    # ---------------------------------------------------------------------------------------
    # Add a paragraph_length column to et_data_enriched
    print("Adding paragraph_length column...")
    paragraph_only_IA = et_data_enriched_no_reread[
        ["unique_paragraph_id", "IA_ID"]
    ].drop_duplicates()
    # paragraph_len_dict is a dictionary that maps from unique_paragraph_id to the number of rows it has in paragraph_only_IA (number of interest areas in the paragraph)
    paragraph_len_df = paragraph_only_IA.groupby("unique_paragraph_id").size()
    # rename the value column to 'paragraph_len'
    paragraph_len_df = paragraph_len_df.rename("paragraph_len").reset_index()
    # join the paragraph_len_df to et_data_enriched
    et_data_enriched = et_data_enriched.join(
        paragraph_len_df.set_index("unique_paragraph_id"), on="unique_paragraph_id"
    )
    # ---------------------------------------------------------------------------------------
    print("Splitting unique_paragraph_id to original columns")
    et_data_enriched["batch"] = (
        et_data_enriched["unique_paragraph_id"]
        .apply(lambda x: x.split("_")[0])
        .astype(int)
    )
    et_data_enriched["article_id"] = (
        et_data_enriched["unique_paragraph_id"]
        .apply(lambda x: x.split("_")[1])
        .astype(int)
    )
    et_data_enriched["level"] = et_data_enriched["unique_paragraph_id"].apply(
        lambda x: x.split("_")[2]
    )
    et_data_enriched["paragraph_id"] = (
        et_data_enriched["unique_paragraph_id"]
        .apply(lambda x: x.split("_")[3])
        .astype(int)
    )
    et_data_enriched["list"] = (
        et_data_enriched["subject_id"].apply(lambda x: x.split("_")[0][1:]).astype(int)
    )
    
    # has_preview_reread will be the pair of has_preview and has_reread concatenated by _
    et_data_enriched["has_preview_reread"] = (
        et_data_enriched["has_preview"].astype(str)
        + ", "
        + et_data_enriched["reread"].astype(str)
    )
    # ---------------------------------------------------------------------------------------
    print("Adding add_first_read_out_of_2_column indicator column...")
    et_data_enriched = add_first_read_out_of_2_column(et_data_enriched)
    # ---------------------------------------------------------------------------------------
    print("Adding experiment_IA_numbering indicator column...")
    # Add a column 'experiment_IA_numbering' to et_data_enriched
    experiment_IA_numbering_df = (
        et_data_enriched.groupby("subject_id")
        .apply(add_experiment_IA_numbering)
        .reset_index()
        .drop("level_1", axis=1)
    )
    et_data_enriched = et_data_enriched.merge(
        experiment_IA_numbering_df,
        on=["subject_id", "article_ind", "IA_ID"],
        how="left",
    )

    print("Adding paragraph_numbering column...")
    et_data_enriched = add_paragraph_numbering(et_data_enriched)

    # ---------------------------------------------------------------------------------------
    print("Adding article_ind_diff_from_first_reading column for articles 11 and 12...")
    diff_cols = [
        "article_ind",
        "paragraph_numbering",
        "experiment_IA_numbering",
        "reread",
    ]
    pivotdf = (
        et_data_enriched[
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

    # for each column in pivotdf: pivotdf[col] = (pivotdf[col][1] - pivotdf[col][0]).rename(col + '_diff_from_first_reading')
    for col in diff_cols:
        pivotdf[col] = pivotdf[col][1] - pivotdf[col][0]
        if col != "reread":
            pivotdf.rename(
                columns={col: col + "_diff_from_first_reading"}, inplace=True
            )

    # remove the secondary column name
    pivotdf.columns = pivotdf.columns.droplevel(1)
    pivotdf = pivotdf.loc[:, ~pivotdf.columns.duplicated()].copy().reset_index()

    et_data_enriched = pd.merge(
        et_data_enriched,
        pivotdf,
        on=["subject_id", "unique_paragraph_id", "IA_ID", "reread"],
        how="left",
    )

    et_data_enriched["article_ind_of_first_reading"] = (
        et_data_enriched["article_ind"]
        - et_data_enriched["article_ind_diff_from_first_reading"]
    )

    et_data_enriched["paragraph_numbering_of_first_reading"] = (
        et_data_enriched["paragraph_numbering"]
        - et_data_enriched["paragraph_numbering_diff_from_first_reading"]
    )

    et_data_enriched["experiment_IA_numbering_of_first_reading"] = (
        et_data_enriched["experiment_IA_numbering"]
        - et_data_enriched["experiment_IA_numbering_diff_from_first_reading"]
    )
    # ---------------------------------------------------------------------------------------

    # Words that were not fixated appear as '.'. Convert them to nan
    et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"] = et_data_enriched[
        "IA_REGRESSION_OUT_FULL_COUNT"
    ].replace(".", np.nan)
    # turn the rest of the column to numeric
    et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"] = pd.to_numeric(
        et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"]
    )

    # Create a column IA_FIRST_PASS_GAZE_DURATION which is IA_FIRST_RUN_DWELL_TIME where IA_FIRST_FIX_PROGRESSIVE  == 1 and NaN otherwise
    et_data_enriched["IA_FIRST_PASS_GAZE_DURATION"] = et_data_enriched[
        "IA_FIRST_RUN_DWELL_TIME"
    ]
    et_data_enriched.loc[
        et_data_enriched["IA_FIRST_FIX_PROGRESSIVE"] != 1, "IA_FIRST_PASS_GAZE_DURATION"
    ] = np.nan
    
    et_data_enriched["IA_ZERO_TF"] = et_data_enriched["IA_DWELL_TIME"] == 0
    
    print("""Adding variables indicating rereading trials with the
          same / different critical span as in the first reading""".replace('\n', ''))
    et_data_enriched = adding_same_diff_critical_span_vars(et_data_enriched)

    return et_data_enriched


def main():
    print("Loading data and removing NaN columns...")
    data = pd.read_csv(
        "data/interim/ia_data_enriched_yoav_cogsci_all_participants.csv"
    )

    et_data_enriched = process_df_for_reread_analysis(data)

    saving_path = (
        data_path + "/interim/et_data_for_reread_analysis_all_participants.csv"
    )
    print("Saving data to...")
    print(saving_path)
    et_data_enriched.to_csv(saving_path, index=False)


if __name__ == "__main__":
    main()
