import logging
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Dict, Any

import censusdis.data as ced
import numpy as np
import pandas as pd
import shap
import xgboost
import yaml
from impactchart.model import XGBoostImpactModel, KnnImpactModel
from censusdis.datasets import ACS5
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.spatial.distance import minkowski

import rih.util as util
from rih.loggingargparser import LoggingArgumentParser
from rih.util import xyw, read_data

logger = logging.getLogger(__name__)


def shap_force(
    xgb_params: Dict[str, Any],
    gdf_cbsa_bg: pd.DataFrame,
    year: int,
    group_lh_together: bool,
    random_state: int,
):
    # Sample randomly before training.
    gdf_cbsa_bg = gdf_cbsa_bg.sample(frac=0.8, random_state=random_state).reset_index(
        names="original_index"
    )
    np.random.seed(random_state)

    X, w, y = xyw(gdf_cbsa_bg, year, group_lh_together)

    logger.info(f"Instantiating XGB with {xgb_params}.")

    # Use the optimal hyperparams, but fit on all the data.
    xgb = xgboost.XGBRegressor(eval_metric="rmsle", **xgb_params)
    xgb = xgb.fit(X=X, y=y, sample_weight=w)

    y_hat = xgb.predict(X=X)

    explainer = shap.TreeExplainer(xgb)

    shap_values = explainer.shap_values(X)

    expected_value = explainer.expected_value

    df_forces = pd.DataFrame(
        shap_values,
        columns=X.columns,
        index=X.index,
    )

    # TODO - verify forces sum correctly.

    df_forces = df_forces.rename(
        {force_col: f"SHAP_{force_col}" for force_col in X.columns}, axis="columns"
    )

    for force_col in X.columns:
        df_forces[force_col] = X[force_col]
        df_forces[f"rel_SHAP_{force_col}"] = df_forces[f"SHAP_{force_col}"] / y_hat

    df_forces["y_hat"] = y_hat
    df_forces["expected_value"] = expected_value
    df_forces["original_index"] = gdf_cbsa_bg["original_index"]
    df_forces[w.name] = w
    df_forces["random_state"] = random_state

    return df_forces


once = False


def main():
    parser = LoggingArgumentParser(logger)

    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument("--group-hispanic-latino", action="store_true")

    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        choices=["xgb", "knn"],
        default="xgb",
        help="Note: knn is experimental and performance is currently terrible. "
        "Avoid unless you are actively developing improvements.",
    )

    parser.add_argument(
        "-t",
        "--tree-param_file",
        help="Tree parameter file, as created by treegress.py",
    )

    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for plots."
    )

    parser.add_argument("-r", "--add_relative", default=True, action="store_true")

    parser.add_argument("--background", default=False, action=BooleanOptionalAction)
    parser.add_argument("--bounds", default=False, action=BooleanOptionalAction)

    parser.add_argument("input_file", help="Input file, as created by datagen.py")

    args = parser.parse_args()

    output_dir = args.output_dir

    if args.model_type == "xgb":
        with open(args.tree_param_file) as f:
            result = yaml.full_load(f)

        params = result["params"]
    elif args.model_type == "knn":
        params = {}
    else:
        logger.warning(f"Unrecognized model type {args.model_type}.")

    gdf_cbsa_bg = read_data(args.input_file, drop_outliers=True)

    print(gdf_cbsa_bg.columns)

    n = len(gdf_cbsa_bg.index)
    k = 5  # 50
    seed = 0x6A1C55E7

    # Median household income in last 12 months.
    # See https://api.census.gov/data/2021/acs/acs5/groups/B19013.html
    income_features = ["B19013_001E"]

    # Fractional demographic features go into the X.
    # Note that we use the ones that are in the data set,
    # which are only the leaves of the group and the single
    # non-leaf frac_B03002_012E which aggregates all Hispanic
    # or Latino residents without regard to race.
    # See https://api.census.gov/data/2021/acs/acs5/groups/B03002.html
    fractional_demographic_features = [
        feature
        for feature in gdf_cbsa_bg.columns
        if feature.startswith("frac_B03002_")
        and "frac_B03002_003E" <= feature <= "frac_B03002_012E"
    ]

    x_features = fractional_demographic_features + income_features

    # Median home value.
    # See https://api.census.gov/data/2021/acs/acs5/groups/B25077.html
    y_col = "B25077_001E"

    if args.model_type == "xgb":
        impact_model = XGBoostImpactModel(
            ensemble_size=k, random_state=seed, estimator_kwargs=params
        )
        # Total owner-occupied households.
        # See https://api.census.gov/data/2020/acs/acs5/groups/B25003.html
        sample_weight_col = "B25003_002E"

        sample_weight = gdf_cbsa_bg[sample_weight_col]
    else:
        weights = np.ones(len(x_features))
        # We scale down the median income by the max
        # value it has in the census.
        weights[-1] = 1 / 250_000.0

        def _income_scaled_minkowski(u, v):
            return minkowski(u, v, p=2, w=weights)

        impact_model = KnnImpactModel(
            ensemble_size=k,
            random_state=seed,
            estimator_kwargs={"metric": _income_scaled_minkowski},
        )
        sample_weight = None

    impact_model.fit(
        gdf_cbsa_bg[x_features], gdf_cbsa_bg[y_col], sample_weight=sample_weight
    )

    logger.info("Generating impact charts.")

    plots = impact_model.impact_charts(
        gdf_cbsa_bg[x_features],
        x_features,
        subplots_kwargs=dict(
            figsize=(12, 8),
        ),
    )

    year = args.vintage

    for feature in x_features:
        logger.info(f"Plotting {feature}.")

        name = Path(output_dir).parent.name.replace("_", " ")
        plot_id = _plot_id(feature, k, n, seed)

        fig, ax = plots[feature]

        label = _style_impact_chart(ax, feature, year, name, plot_id)

        filename = label.replace(" ", "-").replace(";", "")
        logger.info(f"Saving output to {filename}")
        fig.savefig(Path(output_dir) / f"{filename}.png")


def _style_impact_chart(ax, feature, year, name, plot_id):
    """Add style to the basic impact chart the library gives us."""
    col_is_fractional = feature.startswith("frac_")

    # All the variables in both groups that become a part of our X.
    all_variables = pd.concat(
        [
            ced.variables.all_variables(ACS5, year, util.GROUP_RACE_ETHNICITY),
            ced.variables.all_variables(ACS5, year, util.GROUP_MEDIAN_INCOME),
        ]
    )

    feature_base = feature.replace("frac_", "")

    label = all_variables[all_variables["VARIABLE"] == feature_base]["LABEL"].iloc[0]
    label = label.replace("Estimate!!", "")
    label = label.replace("Total:!!", "")
    label = label.replace(":!!", "; ")
    label = label.replace(":", "")

    if col_is_fractional:
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))

    dollar_formatter = FuncFormatter(
        lambda d, pos: f"\\${d:,.0f}" if d >= 0 else f"(\\${-d:,.0f})"
    )

    if col_is_fractional:
        x_width = 1.0
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        x_width = util.MAX_INCOME
        ax.xaxis.set_major_formatter(dollar_formatter)

    ax.yaxis.set_major_formatter(dollar_formatter)

    ax.set_title(f"Impact of {label}\non Median Home Value\n{name}")
    ax.set_xlabel(label)
    ax.set_ylabel("Impact")

    ax.text(
        0.99,
        0.01,
        plot_id,
        fontsize=8,
        backgroundcolor="white",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlim(-0.05 * x_width, 1.05 * x_width)
    ax.axhline(0, color="black", zorder=1)
    ax.grid()

    return label


def _plot_id(feature, k, n, seed):
    return f"(f = {feature}; n = {n:,.0f}; k = {k}; s = {seed:08X})"


if __name__ == "__main__":
    main()
