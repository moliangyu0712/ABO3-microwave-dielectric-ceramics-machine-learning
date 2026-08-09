# xgb_hp.py  
import os  
import pandas as pd  
from pathlib import Path  
  
  
# GetProject rootPathAndOutputPath  
CURRENT_FILE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", CURRENT_FILE_DIR.parent.parent))   
  
  
# S3 OutputTotalPath  
S3_OUTPUT_DIR = Path(os.environ.get("S3_OUTPUT_DIR", BASE_DIR / "output" / "S3-feature_selection"))   
  
  
# XGB SuperParametersOutputSourceFileFolder (hp)   
dire = os.environ.get("XGB_HP_OUTPUT_DIR", str(S3_OUTPUT_DIR / "xgb" / "hp"))   
output_xgb_dir = Path(os.environ.get("XGB_OUTPUT_DIR", S3_OUTPUT_DIR / "xgb"))   
  
  
# FeatureMappingTable  
feature_comb_file = S3_OUTPUT_DIR / "feature_combination.csv"   
importance_combination = pd.read_csv(feature_comb_file)   
  
  
# FromEnvironmentVariableGetSuperParametersRange  
max_depth_str = os.environ.get("XGB_MAX_DEPTH_VALUES", "2,4,6,8,10")   
learning_rate_str = os.environ.get("XGB_LEARNING_RATE_VALUES", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")   
  
  
max_depth_list = [int(x) for x in max_depth_str.split(",")]   
learning_rate_list = [float(x) for x in learning_rate_str.split(",")]   
  
  
cv_repeats = int(os.environ.get("XGB_CV_REPEATS", 50))   
  
  
# UsingColumnTableCollect dict to replaceLowEffective DataFrame._append  
r2_te_list = []   
kk = 0  
  
  
# EstablishDynamic Index and Columns  
init_index = list(range(len(max_depth_list)))   
init_columns = [f"V{i}" for i in range(1, len(learning_rate_list) + 1)]   
  
  
print(f"StartAggregateAndSearchFindEachFeatureCombinationCorrectCorresponding XGBoost OptimalTuningSuperParameters (Total {len(importance_combination)} Group)...")   
  
  
for i in importance_combination.values.tolist():   
    r2_te_sum = pd.DataFrame(0.0, index=init_index, columns=init_columns)   
    kk += 1  
      
    # AccumulateComputeAverageValue  
    for j in range(1, cv_repeats + 1):   
        # Compatible R GenerateFileNameInFrontPlaceEmptyFormat  
        name = f' {j} {i[0]} r2_test.csv'   
        file_path = os.path.join(dire, name)   
          
        if not os.path.exists(file_path):   
            raise FileNotFoundError(f"NotFoundCorrectCorrespondingCrossverificationResultFile: {file_path}")   
              
        file = pd.read_csv(file_path)   
          
        # AutoSortColumnWrongPosition(IfContainsHasAuto-incrementColumnTable)   
        if file.shape[1] > len(learning_rate_list):   
            file = file.iloc[:, 1:]   
              
        r2_te_sum = r2_te_sum + file  
          
    r2_te_sum = r2_te_sum / cv_repeats  # ComputeaverageValue  
      
    # RebuildRowIndexAndColumnNameAsCorrectCorrespondingParametersValue  
    r2_te_sum.index = max_depth_list  
    r2_te_sum.columns = learning_rate_list  
      
    # --- OptimizeCheckFindLogic: Using stack + idxmax FastQuick lockFirstMaximumValue (max_depth, eta) Coordinate ---
    stacked = r2_te_sum.stack()
    best_max_depth, best_eta = stacked.idxmax()  # DefaultAutoFilterOutputFirstMaximumValueCorrectCorresponding(row_index, col_name)
    fc_max_r2_te = stacked[(best_max_depth, best_eta)]
    
    # RecordPerOneFeatureCombinationOptimalSuperParameters
    r2_te_list.append({   
        'features': i[0],   
        'best_hp_result': fc_max_r2_te,   
        'max_depth': best_max_depth,   
        'eta': best_eta  
    })  
  
  
# ColumnTableOneSingle-useConvertAs DataFrame
r2_te = pd.DataFrame(r2_te_list)

# EnsureOutput CSV ColumnWithHasDetermineOrder  
r2_te = r2_te[['features', 'best_hp_result', 'max_depth', 'eta']]   
output_csv = output_xgb_dir / 'r2_te.csv'   
r2_te.to_csv(output_csv, index=False)   
print(f"✅ Optimal XGB RowParametersExtractMappingTableAlreadySave: {output_csv}")