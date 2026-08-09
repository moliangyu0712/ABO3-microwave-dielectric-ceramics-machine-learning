# result_processing.py
import os
import sys
import pandas as pd

# Display all columnsandRow
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ReadEnvironmentVariable
model_dir = os.environ.get("S4_MODEL_DIR")
formula_str = os.environ.get("S4_FORMULA")
target_col = os.environ.get("TARGET", "targ")

if not model_dir or not formula_str:
    print("Error: S4_MODEL_DIR or S4_FORMULA EnvironmentVariableMissing")
    sys.exit(1)

def load_csv_robust(directory, filename):
    p1 = os.path.join(directory, filename)
    p2 = os.path.join(directory, f" {filename}")
    if os.path.exists(p1):
        return pd.read_csv(p1)
    elif os.path.exists(p2):
        return pd.read_csv(p2)
    else:
        raise FileNotFoundError(f"FindNotToNecessaryResultFile (CheckEmptyFormat): {p1}")

# GetAlreadyTrimFeatureInputdata
input_csv_path = os.path.join(model_dir, f"{formula_str}.csv")
data = pd.read_csv(input_csv_path)

# ReadMultiple repeatsBottomPredictionResult
r2_test = load_csv_robust(model_dir, "r2_test.csv")
r2_train = load_csv_robust(model_dir, "r2_train.csv")
rmse_test = load_csv_robust(model_dir, "rmse_test.csv")
rmse_train = load_csv_robust(model_dir, "rmse_train.csv")
pred_test = load_csv_robust(model_dir, "pred_test.csv")
pred_train = load_csv_robust(model_dir, "pred_train.csv")

y = data[target_col] # ExperimentValue
test_data = pred_test.iloc[:, 1:]
train_data = pred_train.iloc[:, 1:]

L = len(y)
# ComputePredictionaverageValueandStandardPoor
Y_test_mean = test_data.mean(axis=1)
Y_train_mean = train_data.mean(axis=1)
Y_test_std = test_data.std(axis=1)
Y_train_std = train_data.std(axis=1)

# AE Calculate
AE_test = pd.DataFrame()
AE_train = pd.DataFrame()
for i in range(0, L):
    name = f'K{i}'
    AE_test[name] = test_data.iloc[i, :] - y.iloc[i]
AE_test = AE_test.T
AE_test_mean = AE_test.mean(axis=1).to_frame()
AE_test_mean.reset_index(inplace=True)

for i in range(0, L):
    name = f'K{i}'
    AE_train[name] = train_data.iloc[i, :] - y.iloc[i]
AE_train = AE_train.T
AE_train_mean = AE_train.mean(axis=1).to_frame()
AE_train_mean.reset_index(inplace=True)

# Absolute deviationCorrectValue
xigema_test = pd.DataFrame()
xigema_train = pd.DataFrame()
for i in range(0, L):
    name = f'K{i}'
    xigema_test[name] = (abs(test_data.iloc[i, :] - y.iloc[i])) / y.iloc[i]
xigema_test = xigema_test.T
ypred_test_mean = xigema_test.mean(axis=1).to_frame()
ypred_test_mean.reset_index(inplace=True)

for i in range(0, L):
    name = f'K{i}'
    xigema_train[name] = (abs(train_data.iloc[i, :] - y.iloc[i])) / y.iloc[i]
xigema_train = xigema_train.T
ypred_train_mean = xigema_train.mean(axis=1).to_frame()
ypred_train_mean.reset_index(inplace=True)

result = pd.DataFrame()
result['targ'] = y
result['train_mean'] = Y_train_mean
result['test_mean'] = Y_test_mean
result['train_std'] = Y_train_std
result['test_std'] = Y_test_std
result['train_xigema'] = ypred_train_mean[0]
result['test_xigema'] = ypred_test_mean[0]
result['train_AE'] = AE_train_mean[0]
result['test_AE'] = AE_test_mean[0]

output_csv = os.path.join(model_dir, "result_data.csv")
result.to_csv(output_csv, index=False)
print(f"✅ S4 Fit single instance metric collectionTotalCompleted: {output_csv}")