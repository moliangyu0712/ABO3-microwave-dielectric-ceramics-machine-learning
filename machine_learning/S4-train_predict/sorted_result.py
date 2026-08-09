# sorted_result.py
import os
import sys
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='svr, xgb, rf, or gbdt')
    args = parser.parse_args()
    
    model = args.model.lower()
    
    project_root = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    s3_output_dir = os.environ.get("S3_OUTPUT_DIR", os.path.join(project_root, "output", "S3-feature_selection"))
    s4_output_dir = os.environ.get("S4_OUTPUT_DIR", os.path.join(project_root, "output", "S4-train_predict"))
    
    # GetFeatureSelectOptimalFeatureResultPath
    model_s3_dir = os.path.join(s3_output_dir, model)
    s3_csv_path = os.path.join(model_s3_dir, "final_sample_average_metrics.csv")
    
    if not os.path.exists(s3_csv_path):
        print(f"Error: FindNotToCorrectCorresponding S3 MetricFile: {s3_csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(s3_csv_path)
    
    if "test_r2" not in df.columns or "formula" not in df.columns:
        print("Error: Metric CSV InLacking 'test_r2' or 'formula' Column.")
        sys.exit(1)
        
    # According to test_r2 HighToLowSortAndPreserveFront 5 Item
    top_5 = df.sort_values(by="test_r2", ascending=False).head(5)
    
    # CreateModel S4 OutputRootDirectory
    model_s4_dir = os.path.join(s4_output_dir, model)
    os.makedirs(model_s4_dir, exist_ok=True)
    
    print(f"--- FilterOutput {model}  Top 5 Best R2 FormulaCombination ---")
    for idx, row in top_5.iterrows():
        formula = str(row["formula"])
        test_r2 = row["test_r2"]
        print(f" - {formula} (test_r2: {test_r2})")
        
        # ToFormulaNameCreateFileFolder
        formula_folder = os.path.join(model_s4_dir, formula)
        os.makedirs(formula_folder, exist_ok=True)
        
    # SaveFront5Item detailsFileProvideBackContinue Python and R TaskTrack
    top5_csv_path = os.path.join(model_s4_dir, "top_5_formulas.csv")
    top_5.to_csv(top5_csv_path, index=False)
    print(f"✅ Top 5 FeatureFormulaFileFolderAlreadyInitialize,MappingTupledataAlreadySave: {top5_csv_path}")

if __name__ == "__main__":
    main()