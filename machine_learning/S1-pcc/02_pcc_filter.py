import argparse
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def safe_filename(name):
    """
    PreventFeatureNameInsideHas / \\ : * ? " < > | etc.CauseFileNameNonMethod
    """
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))

def run_pcc_filter(
    input_file,
    output_dir,
    target_column,
    threshold=0.8,
    use_abs=False,
    draw_heatmap=True,
    draw_scatter=True
):
    """
    CalculateFeatureBetween Pearson RelatedSystemCountand p Value,AndOutputHighRelatedFeatureCorrect.

    Parameters
    ----------
    input_file : str
        Input CSV FilePath.
    output_dir : str
        OutputFileFolder.
    target_column : str
        TargetVariableColumnName,For example targ.
    threshold : float
        PCC Threshold,For example 0.8.
    use_abs : bool
        IsNoUsingAbsoluteCorrectValueFilter.If True,ThenFilter abs(PCC) > threshold.
        If False,ThenFilter PCC > threshold.
    draw_heatmap : bool
        IsNoPlot heatmap.
    draw_scatter : bool
        IsNoScatter plot drawing.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ===================== 1. Readdata =====================
    data = pd.read_csv(input_file)

    if target_column not in data.columns:
        raise ValueError(f"TargetColumn {target_column} NotIndataIn,PleaseCheck.CurrentColumnNameAs: {list(data.columns)}")

    X = data.drop(columns=[target_column])

    # OnlyPreserveCountValueModelFeature,AvoidStringColumnCause corr or pearsonr ReportWrong
    features = X.select_dtypes(include=[np.number])

    if features.shape[1] == 0:
        raise ValueError("NoDetectToCountValueModelFeature,PleaseCheckInputdata.")

    cols = features.columns

    print(f"InputFile: {input_file}")
    print(f"OutputDirectory: {output_dir}")
    print(f"TargetColumn: {target_column}")
    print(f"FeatureCount: {len(cols)}")
    print(f"PCC Threshold: {threshold}")
    print(f"IsNoUsing abs(PCC): {use_abs}")

    # ===================== 2. Calculate PCC and pValueMatrix =====================
    pcc_matrix = features.corr(method="pearson")

    p_values = np.zeros_like(pcc_matrix, dtype=float)

    for i in range(len(cols)):
        for j in range(len(cols)):
            x = features.iloc[:, i]
            y = features.iloc[:, j]

            # IfCertainColumnIsOftenCountColumn,pearsonr WillGenerateWARNINGor nan
            if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
                p_values[i, j] = np.nan
            else:
                corr, p = pearsonr(x, y)
                p_values[i, j] = p

    p_matrix = pd.DataFrame(p_values, index=cols, columns=cols)

    # SaveComplete PCC Matrixand p ValueMatrix
    pcc_matrix_path = os.path.join(output_dir, "PCC_matrix.csv")
    p_matrix_path = os.path.join(output_dir, "p_value_matrix.csv")

    pcc_matrix.to_csv(pcc_matrix_path, encoding="utf-8-sig")
    p_matrix.to_csv(p_matrix_path, encoding="utf-8-sig")

    # ===================== 3. Plot heatmap =====================
    if draw_heatmap:
        mask = np.triu(np.ones_like(pcc_matrix, dtype=bool))
        combined = np.where(mask, pcc_matrix, p_matrix)
        combined_df = pd.DataFrame(combined, index=cols, columns=cols)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            combined_df,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 8}
        )

        if use_abs:
            title_text = f"PCC Upper | p-value Lower | abs(PCC) > {threshold}"
        else:
            title_text = f"PCC Upper | p-value Lower | PCC > {threshold}"

        plt.title(title_text, fontsize=14)
        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, "PCC_Significance_Heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("HeatmapAlreadySave: ", heatmap_path)

    # ===================== 4. ExportFilterBack CSV =====================
    result_rows = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pcc_val = pcc_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]

            if pd.isna(pcc_val):
                continue

            if use_abs:
                condition = abs(pcc_val) > threshold
            else:
                condition = pcc_val > threshold

            if condition:
                result_rows.append([
                    cols[i],
                    cols[j],
                    round(pcc_val, 4),
                    round(p_val, 6)
                ])

    result_df = pd.DataFrame(
        result_rows,
        columns=["VerticalCoordinateFeature", "HorizontalCoordinateFeature", "PCCValue", "pValue"]
    )

    if use_abs:
        csv_name = f"PCC_abs_above_{threshold}.csv"
    else:
        csv_name = f"PCC_above_{threshold}.csv"

    csv_path = os.path.join(output_dir, csv_name)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("HighRelatedFeatureCorrect CSV AlreadySave: ", csv_path)
    print(f"FilterOutput {len(result_df)} CorrectHighRelatedFeature.")

    # ===================== 5. BatchScatter plot drawing =====================
    if draw_scatter:
        scatter_dir = os.path.join(output_dir, "dataInformation")
        os.makedirs(scatter_dir, exist_ok=True)

        for idx, row in result_df.iterrows():
            feat1 = row["VerticalCoordinateFeature"]
            feat2 = row["HorizontalCoordinateFeature"]
            pcc = row["PCCValue"]
            p_val = row["pValue"]

            plt.figure(figsize=(7, 5))
            sns.scatterplot(
                x=features[feat1],
                y=features[feat2],
                color="#2E86AB",
                alpha=0.7
            )

            plt.title(f"{feat1} vs {feat2}\nPCC = {pcc:.3f} | p = {p_val:.2e}", fontsize=12)
            plt.xlabel(feat1)
            plt.ylabel(feat2)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_name = f"scatter_{safe_filename(feat1)}_{safe_filename(feat2)}.png"
            plt.savefig(os.path.join(scatter_dir, save_name), dpi=300, bbox_inches="tight")
            plt.close()

        print("Scatter plotAlreadySaveTo: ", scatter_dir)

    print("PCC AnalysisCompleted.")

    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Pearson RelatedSystemCountFilterScript")

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV FilePath"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="OutputFileFolderPath"
    )

    parser.add_argument(
        "--target",
        default="targ",
        help="TargetVariableColumnName,Default targ"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="PCC Threshold,Default 0.8"
    )

    parser.add_argument(
        "--use_abs",
        action="store_true",
        help="IsNoUsing abs(PCC) > threshold ProceedFilter"
    )

    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="NotPlot heatmap"
    )

    parser.add_argument(
        "--no_scatter",
        action="store_true",
        help="NotScatter plot drawing"
    )

    args = parser.parse_args()

    run_pcc_filter(
        input_file=args.input,
        output_dir=args.output_dir,
        target_column=args.target,
        threshold=args.threshold,
        use_abs=args.use_abs,
        draw_heatmap=not args.no_heatmap,
        draw_scatter=not args.no_scatter
    )

if __name__ == "__main__":
    main()