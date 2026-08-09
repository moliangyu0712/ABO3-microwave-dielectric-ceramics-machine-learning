import numpy as np
import itertools
import pandas as pd
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set Pandas DisplayOptions
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# GetPathConfiguration
CURRENT_FILE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", CURRENT_FILE_DIR.parent))

# GetInputAndOutputVariable
S3_OUTPUT_DIR = Path(os.environ.get("S3_OUTPUT_DIR", BASE_DIR / "output" / "S3-feature_selection"))
TOP_N_FEATURES = int(os.environ.get("TOP_N_FEATURES", 10))
INPUT_FILE = S3_OUTPUT_DIR / f"selected_top_{TOP_N_FEATURES}_features.csv"



# dataSetInTargetColumnName
TARGET = os.environ.get("TARGET", "targ")

# EnsureOutputDirectoryAnd sub-DirectoryExists
os.makedirs(S3_OUTPUT_DIR, exist_ok=True)
comb_dir = S3_OUTPUT_DIR / "feature_combination"
os.makedirs(comb_dir, exist_ok=True)

# verificationInputFileIsNoExists
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"NotFoundInputFeatureFile: {INPUT_FILE}.PleaseEnsureRun firstRow S2-M1-importance GetOutputThisFeatureFile.")

# ReadFeaturedata
data = pd.read_csv(INPUT_FILE)

# EnsureTargetColumnExists
if TARGET not in data.columns:
    raise KeyError(f"TargetColumn '{TARGET}' InInputdataInNotExists.CurrentdataColumnHas: {list(data.columns)}")

# ExtractFeatureVariableColumnTable
features_list = [col for col in data.columns if col != TARGET]

# 1. GenerateAllFeatureCombinationRelationship,AndPackageSave
feature_combination_summary = pd.DataFrame()
for i in range(1, len(features_list) + 1):
    combinations = list(itertools.combinations(features_list, i))
    for c in combinations:
        s = '+'.join(c)
        comb_name = f"{TARGET}~{s}"
        feature_combination_summary = feature_combination_summary._append({'feature': comb_name}, ignore_index=True)

summary_csv_path = S3_OUTPUT_DIR / "feature_combination.csv"
feature_combination_summary.to_csv(summary_csv_path, index=False)
print(f"✅ AlreadySaveFeatureCombinationAssociationFile: {summary_csv_path}")

# 2. SequentiallyGenerateEachSelfIndependentCombination CSV dataSubset
feature_combination_list = pd.DataFrame()
for i in range(1, len(features_list) + 1):
    combinations = list(itertools.combinations(features_list, i))
    for c in combinations:
        s = '+'.join(c)
        feature_combination_list = feature_combination_list._append({'feature': s}, ignore_index=True)

print(f"PositiveInGenerateCombinationSubset CSV File(Total {len(feature_combination_list)} )...")
for i in feature_combination_list.values.tolist():
    result = pd.DataFrame()
    after_split = i[0].split("+")
    for j in after_split:
        result[j] = data[j]
    result[TARGET] = data[TARGET]
    name = f"{TARGET}~{i[0]}"
    output_path = comb_dir / f"{name}.csv"
    result.to_csv(output_path, index=False)

print(f"✅ AlreadySuccessAllSubsetdataFileOutputToDirectory: {comb_dir}")
print("=" * 60)