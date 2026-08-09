# R2-RMSE-MAE.py
import numpy as np
import pandas as pd
import os
import sys

def calculate_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return 1 - (sse / sst) if sst != 0 else 0

def calculate_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def calculate_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

# GetInputParameters
model_type_dir = os.environ.get("S4_MODEL_TYPE_DIR")
if not model_type_dir:
    print("Error: NotSpecify S4_MODEL_TYPE_DIR EnvironmentVariable.")
    sys.exit(1)

result = pd.DataFrame()

# Scan 5 FormulaFileFolder
for item in os.listdir(model_type_dir):
    item_path = os.path.join(model_type_dir, item)
    if os.path.isdir(item_path):
        result_file = os.path.join(item_path, "result_data.csv")
        if os.path.exists(result_file):
            print(f"CollectTotalEvaluationSubDirectory: {item}")
            data_file = pd.read_csv(result_file)
            
            y = pd.to_numeric(data_file["targ"], errors='coerce') 
            
            y_pred_train = pd.to_numeric(data_file["train_mean"], errors='coerce') 
            r2_train = calculate_r2(y, y_pred_train)
            rmse_train = calculate_rmse(y, y_pred_train)
            mae_train = calculate_mae(y, y_pred_train)

            y_pred_test = pd.to_numeric(data_file["test_mean"], errors='coerce') 
            r2_test = calculate_r2(y, y_pred_test)
            rmse_test = calculate_rmse(y, y_pred_test)
            mae_test = calculate_mae(y, y_pred_test)

            row_data = pd.DataFrame([{
                "formula": item,
                "r2_train": r2_train,
                "rmse_train": rmse_train,
                "mae_train": mae_train,
                "r2_test": r2_test,
                "rmse_test": rmse_test,
                "mae_test": mae_test
            }])
            result = pd.concat([result, row_data], ignore_index=True)

output_eval = os.path.join(model_type_dir, "overall_evaluation.csv")
result.to_csv(output_eval, index=False)
print(f"🎉 SuccessGenerateThisComputeMethodFinalCorrectRatioScoreTable: {output_eval}")