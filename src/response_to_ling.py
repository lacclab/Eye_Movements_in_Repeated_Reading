import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from julia_linear_mm import Normal, run_linear_mm


class config:
    def __init__(self):
        self.method = "MM_coeffs"
        self.outcome_variable = "log_IA_DWELL_TIME"
        self.explanatory_variable: str | list[str] = "reread"
        self.data_path = "/data"

        self.fig_path = "/figs"
        self.exclusion_df: bool = True
        self.model_formula: str = ""
        # self.log_transform_outcome = False
        self.z_score_w_r_t_explanatory = True
        # query is article_ind_of_first_reading >= 0
        self.df_query = ""
        self.re_columns = ["subject_id", "unique_paragraph_id"]

        self.config_name = (
            "Coeffs. " + self.outcome_variable + "_" + self.explanatory_variable
        )


def compare_coeffs_of_2_groups(
    df_input: pd.DataFrame,
    comparison_var: str,
    comparison_var_vals: list,
    outcome_variable: str,
    re_columns: list[str],
    link_dist=Normal(),
):
    assert (
        len(comparison_var_vals) == 2
    ), "comparison_var_vals must be a list of 2 values"

    df = df_input.copy()
    df["indicator"] = df[comparison_var].apply(
        lambda x: 0
        if x == comparison_var_vals[0]
        else 1
        if x == comparison_var_vals[1]
        else np.nan
    )
    # drop nans according to the indicator column
    df = df.dropna(subset=["indicator"])

    ling_featurse_formula_interactions = f"""
    {outcome_variable} ~ 1 + Wordfreq_Frequency * Length + gpt2_Surprisal + prev_Wordfreq_Frequency + prev_gpt2_Surprisal + prev_Length +
        Wordfreq_Frequency * indicator + gpt2_Surprisal * indicator + Length * indicator +
        (1 + Wordfreq_Frequency + gpt2_Surprisal + Length | subject_id) + 
        (1 + Wordfreq_Frequency + gpt2_Surprisal + Length | unique_paragraph_id)
    """

    res = run_linear_mm(
        df.loc[lambda x: x[comparison_var] != np.nan].loc[
            lambda x: x[comparison_var] != "nan"
        ],
        outcome_variable,
        re_columns,
        ling_featurse_formula_interactions,
        model_res_var_name="j_model_mm",
        link_dist=link_dist,
        print_model_res=False,
        centralize_covariates=[
            "Wordfreq_Frequency",
            "gpt2_Surprisal",
            "Length",
            "prev_Wordfreq_Frequency",
            "prev_gpt2_Surprisal",
            "prev_Length",
        ],
        centralize_outcome=True,
    )
    coeffs = res[0]
    coeffs = coeffs.loc[
        lambda x: x["Name"].str.contains("&") and x["Name"].str.contains("indicator")
    ]

    # apply a function on Name that leaves only the first part before the &
    # coeffs['Name'] = coeffs['Name'].apply(lambda x: x.split('&')[0][:-1] if '&' in x else x)

    # to coeffs add a column var_0 and var_1 for the two values of comparison_var_vals
    coeffs["var_0"] = comparison_var_vals[0]
    coeffs["var_1"] = comparison_var_vals[1]
    coeffs["var"] = comparison_var

    return coeffs


# go over all pairs of first_second_reading_types and run a linear mixed model for each pair. Use the function compare_coeffs_of_2_groups
def coeff_diff_stat_test(
    df_input: pd.DataFrame,
    comparison_var: str,
    outcome_variable: str,
    re_columns: list[str],
    link_dist=Normal(),
):
    """This function runs a linear mixed model for each pair of values in comparison_var and returns a dataframe with the coefficients and p-values of the difference between the coefficients of the two groups.

    Args:
        df_input (pd.DataFrame): A dataframe with the data
        comparison_var (str): A column name in df_input
        outcome_variable (str): A column name in df_input
        re_columns (list[str]): A list of column names in df_input which are the random effects
        link_dist (_type_, optional): Defaults to Normal().

    Returns:
        coeff_diff_test_df (pd.DataFrame): A dataframe with the coefficients and p-values of the difference between the coefficients of the two groups.
        signif_coeffs (pd.DataFrame): A dataframe with the coefficients and p-values of the difference between the coefficients of the two groups where the p-value is smaller than 0.05.
    """
    reading_types = df_input[comparison_var].unique()
    pairs = list(itertools.combinations(reading_types, 2))

    df = df_input.copy()

    coeff_diff_test_df_list = []
    for pair in tqdm(pairs):
        if str(pair[0]) not in [np.nan, "nan"] and str(pair[1]) not in [np.nan, "nan"]:
            coeff_diff_test_df_list.append(
                compare_coeffs_of_2_groups(
                    df, comparison_var, pair, outcome_variable, re_columns
                )
            )
    coeff_diff_test_df = pd.concat(coeff_diff_test_df_list)
    signif_coeffs = coeff_diff_test_df.loc[lambda x: x["Pr(>|z|)"] < 0.05]
    return coeff_diff_test_df, signif_coeffs
