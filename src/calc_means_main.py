import pandas as pd
import os
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from julia_linear_mm import Normal, run_linear_mm, Poisson, Bernoulli

# ignore pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'
# set seeds
np.random.seed(42)


def calc_mm_means(
    df: pd.DataFrame,
    outcome_variable: str,
    explanatory_variable: str,
    re_columns: list[str],
    link_dist=Normal(),
    mode: str = "subset_mean",
):
    """
    This function calculates the means of outcome_variable for each value of explanatory_variable

    Args:
        df: the data frame to run the model on
        outcome_variable: the outcome variable name
        explanatory_variable: the explanatory variable name
        re_cols: the random effects columns
        link_dist: the link distribution to use for the model
        mode: the mode to use for calculating the means. Can be either "subset_mean" or "sum_contrast_all_mean". See note
    Returns:
        means_df: a data frame with the means of outcome_variable for each value of explanatory_variable

    Note: - **subset_mean: If I want to compute the mean of reread==1,
            I take the reread column, create a contrast variable which is 1 where reread==1 and 0 otherwise.
            Then I fit the lmm with the formula *outcome ~ 1 + contrast + (1|sub) + (1|item)*.
            The final mean is the intercept + the slope**
            - **sum_contrast_all_mean:  If I want to compute the mean of reread==1,
            I take the subset of the dataset where reread==1,
            Then I fit the lmm with the formula *outcome ~ 1 + (1|sub) + (1|item).*
            The final mean is the intercept**
    """
    assert mode in ["subset_mean", "sum_contrast_all_mean"]
    # remove rows where explanatory_variable is nan or 'nan'
    df = df.loc[lambda x: ~pd.isnull(x[explanatory_variable])]
    df = df.loc[lambda x: x[explanatory_variable] != "nan"]

    u_vals = df[explanatory_variable].unique()
    # remove nan from u_vals
    u_vals = sorted(u_vals[~pd.isnull(u_vals)])

    print(f"u_vals: {u_vals}")

    means_dict = {}
    for val in tqdm(u_vals):
        if mode == "subset_mean":
            subset_df = df[df[explanatory_variable] == val]
            df_for_lmm = subset_df
            formula = f"{outcome_variable} ~ 1 " + " ".join(
                [f" + (1 | {x})" for x in re_columns]
            )
        if mode == "sum_contrast_all_mean":
            raise NotImplementedError
            # df["sum_contrast"] = df[explanatory_variable].apply(
            #     lambda x: 1 if x == val else -1
            # )
            # df_for_lmm = df
            # formula = f"{outcome_variable} ~ 1 + sum_contrast + (1 | subject_id) + (1 | unique_paragraph_id)"

        coeff_table, _ = run_linear_mm(
            df_for_lmm,
            outcome_variable,
            re_columns,
            formula,
            model_res_var_name="j_model",
            link_dist=link_dist,
            centralize_covariates=False,
            centralize_outcome=False,
            z_outcome=False,
            print_model_res=False,
        )
        mean = round(coeff_table["Coef."].sum(), 3)  # beta_0 + beta_1
        if mode == "subset_mean":
            p_val = coeff_table["Pr(>|z|)"].values[0]
        elif mode == "sum_contrast_all_mean":
            raise NotImplementedError
            # p_val = coeff_table["Pr(>|z|)"].values[
            #     1
            # ]  # Hypothesis: the mean of the outcome for this value of the explanatory variable is different from the grand mean
        se = round(
            (coeff_table["Std. Error"].sum() * 1.96),
            3,  # this is actually a single value since the formula supports only 'y ~ 1 + re_terms'
        )
        means_dict[val] = {"mean": mean, "p_val": p_val, "2se": se}
    means_df = (
        pd.DataFrame(means_dict)
        .T.reset_index()
        .rename(columns={"index": "explanatory_variable_value"})
    )
    # To means_df add a column which is always outcome_variable
    means_df["outcome_variable"] = outcome_variable
    means_df["explanatory_variable"] = explanatory_variable
    return means_df


def choose_link_dist(outcome_variable):
    if outcome_variable == "IA_SKIP":
        link_dist = Bernoulli()
    elif outcome_variable in [
        "IA_RUN_COUNT",
        "IA_FIXATION_COUNT",
        "IA_REGRESSION_OUT_FULL_COUNT",
    ]:
        link_dist = Poisson()
    else:
        link_dist = Normal()
    return link_dist


def calc_mm_means_for_all_outcomes(
    df: pd.DataFrame,
    explanatory_variable_list: list[str],
    re_columns: list[str],
    outcomes: list[str],
    mean_mode: str = "subset_mean",
):
    """
    This function calculates the means of outcome_variable for each value of explanatory_variable

    Args:
        df: the data frame to run the model on
        outcome_variable: the outcome variable name
        explanatory_variable: the explanatory variable name
        re_cols: the random effects columns
        link_dist: the link distribution to use for the model

    Returns:
        means_df: a data frame with the means of outcome_variable for each value of explanatory_variable
    """
    means_dfs_lst = []
    df["explanatory_variable"] = df[explanatory_variable_list].apply(
        lambda x: "__".join([str(y) for y in x]), axis=1
    )
    for outcome_variable in outcomes:
        df_m = df.copy()
        # in df_m create a column from explanatory_variable_list which is the concatenation of all explanatory variables devided by __

        # drop all rows where outcome_variable is nan
        # print the number of rows dropped
        print(f"outcome_variable: {outcome_variable}")
        print(f"df_m shape before dropping nan: {df_m.shape}")
        df_m = df_m[~pd.isnull(df_m[outcome_variable])]
        print(f"df_m shape after dropping nan: {df_m.shape}")

        means_dfs_lst.append(
            calc_mm_means(
                df_m,
                outcome_variable,
                "explanatory_variable",
                re_columns,
                mode=mean_mode,
            )
        )

    means_df = pd.concat(means_dfs_lst)

    # now split the column "explanatory_variable" to the different explanatory variable according to explanatory_variable_list
    for i, explanatory_variable in enumerate(explanatory_variable_list):
        means_df[explanatory_variable] = means_df["explanatory_variable_value"].apply(
            lambda x: x.split("__")[i]
        )
    # delete the column "explanatory_variable"
    means_df.drop(
        ["explanatory_variable_value", "explanatory_variable"], axis=1, inplace=True
    )

    return means_df


def plot_means(
    means_all_outcomes: pd.DataFrame,
    explanatory_variable_list: str,
    outcome_name_mapping: dict,
    outcome_units_mapping: dict,
    z_score_outcome: bool = False,
    explanatory_var_class_names: dict = None,
    explanatory_var_names: dict = None,
    update_layout_kwargs: dict = None,
    update_xaxes_kwargs: dict = None,
    update_subplot_layout_kwargs: dict = None,
    save_fig_to_path: str = None,
):
    # Create subplots
    num_cols = len(means_all_outcomes["outcome_variable"].unique()) // 2 + 1
    num_rows = 2 if len(means_all_outcomes["outcome_variable"].unique()) > 4 else 1
    subplot_titles = [
        outcome_name_mapping[x] for x in means_all_outcomes["outcome_variable"].unique()
    ]
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )
    fig.print_grid()

    if explanatory_var_names is not None:
        # rename the columns in the means_all_outcomes df that are in explanatory_var_names
        means_all_outcomes.rename(columns=explanatory_var_names, inplace=True)
        # change the explanatory_variable_list to the new names
        explanatory_variable_list = [
            explanatory_var_names[x] if x in explanatory_var_names.keys() else x
            for x in explanatory_variable_list
        ]

    # Iterate over each outcome_variable
    for i, outcome_variable in enumerate(
        means_all_outcomes["outcome_variable"].unique()
    ):
        row = i // num_cols + 1
        col = i % num_cols + 1
        print(f"row: {row}, col: {col}")
        # Filter the data for the current outcome_variable
        data = means_all_outcomes[
            means_all_outcomes["outcome_variable"] == outcome_variable
        ]
        if outcome_variable in outcome_name_mapping:
            outcome_variable_name = outcome_name_mapping[outcome_variable]
        if z_score_outcome and outcome_variable not in ["IA_SKIP", "IA_RUN_COUNT"]:
            outcome_variable_name = f"{outcome_variable_name} (z-score)"

        for ev in explanatory_variable_list:
            if ev in explanatory_var_class_names.keys():
                # translate the explanatory variable values to the class names
                data[ev] = data[ev].apply(lambda x: explanatory_var_class_names[ev][x])
        colors = [
            "#005CAB",
            "#AF0038",
            "#003366",
            "darkolivegreen",
            "darkcyan",
            "darkslategray",
            "darkslateblue",
            "turquoise",
            "darkslategray",
            "cadetblue",
            "steelblue",
            "royalblue",
            "midnightblue",
        ]
        # remove form the data rows that contain "nan" in explanatory_variable_list[0]
        data = data.loc[lambda x: x[explanatory_variable_list[0]] != "nan"]
        # use isnumeric to turn the values of the explanatory variables to numeric
        data[explanatory_variable_list[0]] = pd.to_numeric(
            data[explanatory_variable_list[0]], errors="ignore"
        )

        # Add bar graph to the subplot
        if data[explanatory_variable_list[0]].nunique() <= 150:
            if len(explanatory_variable_list) == 2:
                for j, secondary_ev_val in enumerate(
                    list(data[explanatory_variable_list[1]].unique())
                ):
                    if secondary_ev_val == "nan":
                        continue
                    data_secondary_ev = data[
                        data[explanatory_variable_list[1]] == secondary_ev_val
                    ]
                    fig.add_trace(
                        go.Bar(
                            x=data_secondary_ev[explanatory_variable_list[0]],
                            y=data_secondary_ev["mean"],
                            error_y=dict(type="data", array=data_secondary_ev["2se"]),
                            name=secondary_ev_val if i == 0 else "",
                            marker=dict(color=colors[j]),
                            legendgroup=outcome_variable_name,
                            showlegend=True if i == 0 else False,
                        ),
                        row=row,
                        col=col,
                    )
            elif len(explanatory_variable_list) == 1:
                fig.add_trace(
                    go.Bar(
                        name="",
                        x=data[explanatory_variable_list[0]],
                        y=data["mean"],
                        error_y=dict(type="data", array=data["2se"]),
                        marker=dict(color=colors[0]),
                        # text=data["mean"],
                        # textposition='auto',
                    ),
                    row=row,
                    col=col,
                )
            else:
                raise ValueError(
                    "explanatory_variable_list can only be of length 1 or 2"
                )

        xaxes_kwargs = {
            "title_text": explanatory_variable_list[0],
            "row": row,
            "col": col,
        }
        # remove from xaxes_kwargs the keys that are in update_xaxes_kwargs
        for k in update_xaxes_kwargs.keys():
            if k in xaxes_kwargs.keys():
                xaxes_kwargs.pop(k)

        fig.update_xaxes(
            **xaxes_kwargs,
            **(update_xaxes_kwargs if update_xaxes_kwargs is not None else {}),
        )

        subplot_update_layout = {
            "showlegend": True,
            "font": dict(size=26),
        }

        # remove from subplot_update_layout the keys that are in subplot_update_layout_kwargs
        for k in update_subplot_layout_kwargs.keys():
            if k in subplot_update_layout.keys():
                subplot_update_layout.pop(k)

        fig.update_layout(
            **subplot_update_layout,
            **(
                update_subplot_layout_kwargs
                if update_subplot_layout_kwargs is not None
                else {}
            ),
        )

        z_score_units = (
            " z-score"
            if z_score_outcome and outcome_variable not in ["IA_SKIP", "IA_RUN_COUNT"]
            else ""
        )
        fig.update_yaxes(
            title_text=f"{z_score_units} ({outcome_units_mapping[outcome_variable]})",
            row=row,
            col=col,
        )
        # fig.update_yaxes(title_text=f"{data['outcome_variable'].unique()[0]} Mean", row=row, col=col)

    layout_params = {
        "height": 1000,
        "width": 2450,
        "titlefont": dict(size=28),
        "font": dict(size=14),
        "showlegend": True if len(explanatory_variable_list) == 2 else False,
        "legend_title_text": explanatory_variable_list[1]
        if len(explanatory_variable_list) == 2
        else None,
        "title_x": 0.5,
    }

    # remove from layout_params the keys that are in update_layout_kwargs
    for k in update_layout_kwargs.keys():
        if k in layout_params.keys():
            layout_params.pop(k)

    fig.update_layout(
        **layout_params,
        **(update_layout_kwargs if update_layout_kwargs is not None else {}),
    )

    if save_fig_to_path is not None:
        fig.write_image(save_fig_to_path)

    return fig
