import time
import scipy
import pandas as pd
import numpy as np
import warnings
# import Value

print("Starting Julia...")

julia_start = time.time()
import julia
from julia.core import JuliaError
from julia.api import Julia

try:
    jl = Julia(compiled_modules=False)
except JuliaError:
    julia.install()
    jl = Julia(compiled_modules=False)

from julia import Main

Main.eval("using Pkg")


def import_julia_packages(packages: list[str]):
    for package in packages:
        try:
            Main.eval(f"using {package}")
        except:  # noqa: E722
            Main.eval(f'Pkg.add("{package}")')
            Main.eval(f"using {package}")


packages = ["MixedModels", "DataFrames", "Pandas", "Distributions", "StatsModels"]

import_julia_packages(packages)


print("Julia started. in took {:.2f} seconds".format(time.time() - julia_start))

from julia.Distributions import Poisson, Bernoulli, Normal # type: ignore
from julia.MixedModels import ( # type: ignore
    MixedModel,
    aic,
    coeftable,
    raneftables,
    dof,
    predict,
    fitted,
    fit,
)
from julia import Pandas
from julia import Base


Main.run_hybrid_pandas = Main.eval(
    """
        using Pandas
        function run_hybrid_pandas(generic_inputs)
            return Pandas.DataFrame(run_hybrid(generic_inputs))
        end
          """
)

Main.eval(
    """
        function pd_to_df(df_pd)
            df= DataFrames.DataFrame()
            for col in df_pd.columns
                df[!, col] = getproperty(df_pd, col).values
            end
            df
        end
        """
)

Main.eval(
    """
        function table_to_pd(x)
            Pandas.DataFrame(x)
        end
        """
)


def add_CIs_to_coef_df(coef_df: pd.DataFrame, dof: int):
    coef_df_copy = coef_df.copy()
    t_quantile = scipy.stats.t(df=dof).ppf(0.975)
    coef_df_copy["l_conf"] = (
        coef_df_copy["Coef."] - t_quantile * coef_df_copy["Std. Error"]
    )
    coef_df_copy["u_conf"] = (
        coef_df_copy["Coef."] + t_quantile * coef_df_copy["Std. Error"]
    )
    return coef_df_copy


def run_linear_mm(
    df_input: pd.DataFrame,
    outcome_variable: str,
    re_cols: list[str],
    formula: str,  # e.g. "outcome_variable ~ covariate1 + covariate2 + (1|re_col1) + (1|re_col2)"
    print_model_res: bool = True,
    model_res_var_name: str = "j_model",
    link_dist=Normal(),  # Normal(), Poisson(), Bernoulli()
    contrasts_dict=None,  # {"col": ""}
    centralize_covariates: list[str] | str | bool = False,
    centralize_outcome: bool = False,
    z_outcome: bool = False,
):
    """
    Run a linear mixed model in Julia using the MixedModels package
    Args:
    df_input: pd.DataFrame
        The dataframe to use
    outcome_variable: str
        The name of the outcome variable
    re_cols: list[str]
        The names of the random effect columns
    formula: str
        The formula to use for the model
        ***** We assume that the operations and variable names in the formula are separated by spaces *****
    print_model_res: bool
        Whether to print the model results
    model_res_var_name: str
        The name of the variable to store the model results in
    link_dist: Julia object
        Link function to use for the model. Can be one of: Normal(), Poisson(), Bernoulli()
    contrasts_dict: dict
        A dictionary of contrasts to use: {"col": "contrast_type"}
        contrast_type can be one of: https://juliastats.org/StatsModels.jl/stable/contrasts/
            HC: HelmertCoding
            EC: EffectsCoding
            DC: DummyCoding
            SDC: SeqDiffCoding
    centralize_covariates: list[str] | str | bool
        Whether to centralize covariates. If a list of covariate names is given, only those covariates will be centralized
    centralize_outcome: bool
        Whether to centralize the outcome variable
    """
    df = df_input.copy()
    # from the formula, extract all covariate names
    contrast_cols = list(contrasts_dict.keys()) if contrasts_dict is not None else []
    numerical_covariate_names = formula.split(" ") + re_cols
    numerical_covariate_names = list(
        set(
            [
                x
                for x in numerical_covariate_names
                if x in df.columns
                and x != outcome_variable
                and x not in re_cols
                and x
                not in contrast_cols  # We exclude contrast_cols from covariate_names because there is no sense in centralizing them
            ]
        )
    )
    if centralize_covariates in ["all", True]:
        if print_model_res:
            print("Centralizing covariates:")
            print(df[numerical_covariate_names].mean())
        df[numerical_covariate_names] = (
            df[numerical_covariate_names] - df[numerical_covariate_names].mean()
        )
    elif isinstance(centralize_covariates, list):
        if print_model_res:
            print("Centralizing covariates:")
            print(df[centralize_covariates].mean())
        df[centralize_covariates] = (
            df[centralize_covariates] - df[centralize_covariates].mean()
        )
    elif centralize_covariates in ["None", False]:
        pass
    else:
        raise ValueError(
            "centralize_covariates should be 'all', 'None' or a list of covariate names"
        )

    if z_outcome:
        df[outcome_variable] = (
            df[outcome_variable] - df[outcome_variable].mean()
        ) / df[outcome_variable].std()
    elif centralize_outcome:
        if print_model_res:
            print("Centralizing outcome:")
            print(df[outcome_variable].mean())
        df[outcome_variable] = df[outcome_variable] - df[outcome_variable].mean()

    df = df[contrast_cols + numerical_covariate_names + [outcome_variable] + re_cols]

    # if df contains nans trigger a warning
    if df.isna().sum().sum() > 0:
        # warn the columns that contain nans
        warnings.warn(
            "The dataframe contains nans. Records with nans are omitted The following columns contain nans: {}".format(
                df.columns[df.isna().sum() > 0]
            )
        )

    df = df.dropna()

    Main.j_df = Main.pd_to_df(df)
    # Make contrast_cols and re_cols categorical (using Main.eval)
    for col in contrast_cols:
        # if the type of col in df is numeric
        if df[col].dtype in [np.int64, np.float64]:
            Main.col_string = col
            Main.eval(
                """
                using CategoricalArrays
                col_symbol = Symbol(col_string)
                j_df[!, col_symbol] = categorical(j_df[!, col_symbol])
                    """
            )

    if contrasts_dict is not None:
        c_dict_description = [
            ":{x} => {y}".format(x=x, y=y) for x, y in contrasts_dict.items()
        ]
        c_dict_description = ", ".join(c_dict_description)

        Main.eval(
            f"""
                using MixedModels
                HC = HelmertCoding()
                EC = EffectsCoding()
                DC = DummyCoding()
                SDC = SeqDiffCoding()
                
                contrasts = Dict({c_dict_description})
                """
        )

    Main.j_formula = Main.eval(f"@formula({formula})")
    Main.link_dist = link_dist
    if contrasts_dict is not None:
        Main.mm_model_res = Main.eval(
            "fit(MixedModel, j_formula, j_df, link_dist, contrasts=contrasts, progress=false)"
        )
    else:
        Main.mm_model_res = Main.eval(
            "fit(MixedModel, j_formula, j_df, link_dist, progress=false)"
        )

    Main.eval(f"""{model_res_var_name} = mm_model_res""")  # fit the mm model

    mm_coeftable = Pandas.DataFrame(coeftable(getattr(Main, model_res_var_name)))

    ref_dfs = Base.Tuple(
        Base.map(Main.table_to_pd, raneftables(getattr(Main, model_res_var_name)))
    )

    if print_model_res:
        print(Main.mm_model_res)

    mm_dof = dof(getattr(Main, model_res_var_name))
    mm_coeftable = add_CIs_to_coef_df(mm_coeftable, mm_dof)

    return mm_coeftable, ref_dfs
