import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def get_project_root():
    """
    range.py PositionAt MACHINE_LEARNING/S2-M1-importance/range.py
    Therefore parents[1] ThenIs MACHINE_LEARNING
    """
    return Path(__file__).resolve().parents[1]

def min_max_normalize(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_value) / (max_value - min_value)

def read_required_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"NotFoundFile: {path}")
    return pd.read_csv(path)

def get_top_n_from_args_or_env():
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    return int(os.getenv("TOP_N_FEATURES", "10"))

def main():
    project_root = get_project_root()

    data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
    output_dir = Path(os.getenv("OUTPUT_DIR", project_root / "output" / "S2-M1-importance"))
    s3_dir = Path(os.getenv("S3_DIR", project_root / "S3-feature_selection"))
    
    # Newly added: FromEnvironmentVariableRead S3_OUTPUT_DIR,UseAtSaveFilterOutputFeatureOriginal sub-dataSet
    s3_output_dir = Path(os.getenv("S3_OUTPUT_DIR", project_root / "output" / "S3-feature_selection"))

    s3_dir.mkdir(parents=True, exist_ok=True)
    s3_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n_features = get_top_n_from_args_or_env()

    print("========== range.py StartIntegrateFeatureImportance ==========")
    print(f"Project rootDirectory: {project_root}")
    print(f"dataDirectory: {data_dir}")
    print(f"S2 OutputDirectory: {output_dir}")
    print(f"S3 FeatureFilterDirectory(CodeDirectory): {s3_dir}")
    print(f"S3 OutputResultDirectory(dataDirectory): {s3_output_dir}")
    print(f"PreserveFront {top_n_features} Feature")

    rf_im1_path = output_dir / "rf_im1.csv"
    rf_im2_path = output_dir / "rf_im2.csv"
    gbdt_im1_path = output_dir / "gbdt_im1.csv"
    xgb_im1_path = output_dir / "xgb_im1.csv"

    IncNodePurity = read_required_csv(rf_im1_path)
    IncMSE = read_required_csv(rf_im2_path)
    Rel_Inf = read_required_csv(gbdt_im1_path)
    xgb_importance = read_required_csv(xgb_im1_path)

    qf_after_pcc_path = data_dir / "data_after-pcc.csv"
    qf_path = data_dir / "data.csv"

    if qf_after_pcc_path.exists():
        source_data_path = qf_after_pcc_path
    elif qf_path.exists():
        source_data_path = qf_path
    else:
        raise FileNotFoundError(
            f"NotFoundInputdata,Please confirmExistsBelowFileOfOne:\n"
            f"{qf_after_pcc_path}\n"
            f"{qf_path}"
        )

    full_data = pd.read_csv(source_data_path)

    if "targ" not in full_data.columns:
        raise ValueError(f"{source_data_path} InNotFoundTargetColumn targ")

    feature_names = list(full_data.columns[:-1])

    # RF: rf_im1.csv / rf_im2.csv UsuallyOneColumnIsRowName,NeedTo remove
    IncNodePurity_values = IncNodePurity.iloc[:, 1:]
    IncMSE_values = IncMSE.iloc[:, 1:]

    if len(IncNodePurity_values) != len(feature_names):
        raise ValueError(
            f"rf_im1.csv FeatureCountanddataColumnCountNotOneLead to: "
            f"rf_im1={len(IncNodePurity_values)}, data_features={len(feature_names)}"
        )

    if len(IncMSE_values) != len(feature_names):
        raise ValueError(
            f"rf_im2.csv FeatureCountanddataColumnCountNotOneLead to: "
            f"rf_im2={len(IncMSE_values)}, data_features={len(feature_names)}"
        )

    # GBDT: gbdt_im1.csv UsuallyOneColumnIsIndex,NeedTo remove
    Rel_Inf_values = Rel_Inf.iloc[:, 1:]

    if len(Rel_Inf_values) != len(feature_names):
        raise ValueError(
            f"gbdt_im1.csv FeatureCountanddataColumnCountNotOneLead to: "
            f"gbdt_im1={len(Rel_Inf_values)}, data_features={len(feature_names)}"
        )

    # IncNodePurity
    row_mean_1 = IncNodePurity_values.mean(axis=1)
    row_mean_1 = min_max_normalize(row_mean_1)

    IncNodePurity_result = pd.DataFrame({
        "IncNodePurity": row_mean_1,
        "Feature": feature_names
    })

    IncNodePurity_result = IncNodePurity_result.sort_values(
        by="IncNodePurity",
        ascending=False
    )

    IncNodePurity_result.to_csv(
        output_dir / "IncNodePurity.csv",
        index=False
    )

    # %IncMSE
    row_mean_2 = IncMSE_values.mean(axis=1)
    row_mean_2 = min_max_normalize(row_mean_2)

    IncMSE_result = pd.DataFrame({
        "%IncMSE": row_mean_2,
        "Feature": feature_names
    })

    IncMSE_result = IncMSE_result.sort_values(
        by="%IncMSE",
        ascending=False
    )

    IncMSE_result.to_csv(
        output_dir / "IncMSE.csv",
        index=False
    )

    # GBDT Rel_Inf
    row_mean_3 = Rel_Inf_values.mean(axis=1)
    row_mean_3 = min_max_normalize(row_mean_3)

    Rel_Inf_result = pd.DataFrame({
        "Rel_Inf": row_mean_3,
        "Feature": feature_names
    })

    Rel_Inf_result = Rel_Inf_result.sort_values(
        by="Rel_Inf",
        ascending=False
    )

    Rel_Inf_result.to_csv(
        output_dir / "Rel_Inf.csv",
        index=False
    )

    # XGB importance
    if "Feature" not in xgb_importance.columns:
        raise ValueError("xgb_im1.csv InNotFound Feature Column")

    xgb_numeric = xgb_importance.drop(columns=["Feature"], errors="ignore")
    xgb_numeric = xgb_numeric.select_dtypes(include=[np.number])

    if xgb_numeric.shape[1] == 0:
        # CompatibleSome xgb_im1.csv OneColumnAs Feature,BackWill be readIntoStringcase
        temp = xgb_importance.drop(columns=["Feature"], errors="ignore")
        xgb_numeric = temp.apply(pd.to_numeric, errors="coerce").fillna(0)

    row_mean_4 = xgb_numeric.mean(axis=1)
    row_mean_4 = min_max_normalize(row_mean_4)

    xgb_im = pd.DataFrame({
        "Feature": xgb_importance["Feature"],
        "xgb_importance": row_mean_4
    })

    xgb_im = xgb_im.sort_values(
        by="xgb_importance",
        ascending=False
    )

    xgb_im.to_csv(
        output_dir / "xgb_importance.csv",
        index=False
    )

    # ComprehensiveResult
    Feature = IncNodePurity_result.merge(
        IncMSE_result,
        on="Feature",
        how="left"
    )

    Feature = Feature.merge(
        Rel_Inf_result,
        on="Feature",
        how="left"
    )

    Feature = Feature.merge(
        xgb_im,
        on="Feature",
        how="left"
    )

    score_columns = [
        "IncNodePurity",
        "%IncMSE",
        "Rel_Inf",
        "xgb_importance"
    ]

    Feature[score_columns] = Feature[score_columns].fillna(0)

    Feature["total"] = (
        Feature["IncNodePurity"]
        + Feature["%IncMSE"]
        + Feature["Rel_Inf"]
        + Feature["xgb_importance"]
    )

    max_total = Feature["total"].max()

    if max_total == 0:
        Feature["IncNodePurity_p"] = 0
        Feature["%IncMSE_p"] = 0
        Feature["Rel_Inf_p"] = 0
        Feature["xgb_importance_p"] = 0
    else:
        Feature["IncNodePurity_p"] = Feature["IncNodePurity"] / max_total
        Feature["%IncMSE_p"] = Feature["%IncMSE"] / max_total
        Feature["Rel_Inf_p"] = Feature["Rel_Inf"] / max_total
        Feature["xgb_importance_p"] = Feature["xgb_importance"] / max_total

    # Here by total FromLargeToSmallSort,For easy viewing of most importantFeature
    Feature = Feature.sort_values(
        by="total",
        ascending=False
    ).reset_index(drop=True)

    comprehensive_path = output_dir / "comprehensive score of Importance.csv"

    Feature.to_csv(
        comprehensive_path,
        index=False
    )

    print(f"Comprehensive importanceResultAlreadySave: {comprehensive_path}")

    # Based onComprehensive importanceFilterFront n Feature
    selected_features = Feature["Feature"].head(top_n_features).tolist()

    missing_features = [
        feature for feature in selected_features
        if feature not in full_data.columns
    ]

    if missing_features:
        raise ValueError(
            "BelowFeatureExistsAt comprehensive score of Importance.csv,"
            "ButNotExistsFrom originaldataIn:\n"
            + ", ".join(missing_features)
        )

    selected_columns = selected_features + ["targ"]
    selected_data = full_data[selected_columns].copy()

    # SaveTo S3_OUTPUT_DIR In
    s3_output_csv_path = s3_output_dir / f"selected_top_{top_n_features}_features.csv"
    selected_data.to_csv(
        s3_output_csv_path,
        index=False
    )
    print(f"FilterBackdataAlreadySameStepSaveTo S3_OUTPUT_DIR: {s3_output_csv_path}")

    print("========== FeatureFilterCompleted ==========")
    print(f"PreserveFeatureCount: {top_n_features}")
    print(f"PreserveFeature: {selected_features}")

if __name__ == "__main__":
    main()