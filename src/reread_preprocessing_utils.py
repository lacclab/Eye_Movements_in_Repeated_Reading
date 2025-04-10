import multiprocessing
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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
    et_df["same_span_in_FR"] = et_df["same_span_in_FR"].fillna(".")

    # put et_data_enriched['same_span_in_FR'] == '.' were et_data_enriched['has_preview'] == Gathering
    et_df.loc[et_df["has_preview"] == "Gathering", "same_span_in_FR"] = "."

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
    et_df.loc[et_df["article_ind"] < 11, "same_span_in_FR_also_gathering"] = "."

    return et_df
