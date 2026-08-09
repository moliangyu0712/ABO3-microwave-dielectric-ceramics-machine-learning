# crop_dataset.py
import os
import sys
import pandas as pd

def main():
    target_col = os.environ.get("TARGET", "targ")
    formula = os.environ.get("S4_FORMULA")
    s3_output_dir = os.environ.get("S3_OUTPUT_DIR")
    top_n = os.environ.get("TOP_N_FEATURES")
    output_dir = os.environ.get("S4_MODEL_DIR")

    if not all([formula, s3_output_dir, top_n, output_dir]):
        print("Error: EnvironmentVariablePassNotAll,NoneMethodProceeddataColumnSplit.")
        sys.exit(1)

    # ReadAfter PCC/S2 BackSelectGet Top-N LargedataSet
    input_csv = os.path.join(s3_output_dir, f"selected_top_{top_n}_features.csv")
    if not os.path.exists(input_csv):
        print(f"Error: NotFound S3  Top FeatureSet: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)

    # ExtractFormulaRight-side factor
    if "~" in formula:
        lhs, rhs = formula.split("~")
        lhs = lhs.strip()
        rhs = rhs.strip()
    else:
        lhs = target_col
        rhs = formula.strip()

    # ExtractFeatureVariable
    features = [x.strip() for x in rhs.split("+") if x.strip()]
    # Remove excess intercept terms
    features = [x for x in features if x not in ["-1", "1", "0", ""]]

    # NeedNeedPreserveColumn
    cols_to_keep = []
    if lhs in df.columns:
        cols_to_keep.append(lhs)
    else:
        cols_to_keep.append(target_col)

    for f in features:
        if f in df.columns:
            cols_to_keep.append(f)
        else:
            print(f"Warning: Variable '{f}' AbsentInLargedataSetIn.")

    cropped_df = df[cols_to_keep]

    # SaveNameAs: Target~FeatureParameterQuantity.csv
    output_file = os.path.join(output_dir, f"{formula}.csv")
    cropped_df.to_csv(output_file, index=False)
    print(f"✅ ColumndataTrim complete,OutputTo: {output_file}")

if __name__ == "__main__":
    main()