import os  
import pandas as pd  
from pathlib import Path  
  
# GetProject rootPathAndOutputPath  
CURRENT_FILE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", CURRENT_FILE_DIR.parent.parent))  
  
# S3 OutputTotalPath  
S3_OUTPUT_DIR = Path(os.environ.get("S3_OUTPUT_DIR", BASE_DIR / "output" / "S3-feature_selection"))  
  
# SVR SuperParametersOutputSourceFileFolder (hp)  
dire = os.environ.get("SVR_HP_OUTPUT_DIR", str(S3_OUTPUT_DIR / "svr" / "hp"))  
output_svr_dir = Path(os.environ.get("SVR_OUTPUT_DIR", S3_OUTPUT_DIR / "svr"))  
  
# FeatureMappingTable   
feature_comb_file = S3_OUTPUT_DIR / "feature_combination.csv"  
importance_combination = pd.read_csv(feature_comb_file)  
  
# FromEnvironmentVariableGetSuperParametersRange  
gamma_str = os.environ.get("SVR_GAMMA_VALUES", "0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256")  
cost_str = os.environ.get("SVR_COST_VALUES", "0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256")  
gamma_list = [float(x) for x in gamma_str.split(",")]  
cost_list = [float(x) for x in cost_str.split(",")]  
  
cv_repeats = int(os.environ.get("SVR_CV_REPEATS", 50))  
  
# Using List To collect dictionary,ReplaceLowEffectiveAnd deprecated DataFrame._append 
r2_te_list = []  
k = 1  
  
# R DefaultConvertNoColumnNameWhen,V1, V2 etc.AutoAsColumnElement
init_index = list(range(len(gamma_list)))  
init_columns = [f"V{i}" for i in range(1, len(cost_list) + 1)]  
  
print(f"StartAggregateandMatchFeature SVR OptimalSuperParameters (TotalHas {len(importance_combination)} FeatureCombination)...")  
  
for i in importance_combination.values.tolist():  
    r2_te_sum = pd.DataFrame(0.0, index=init_index, columns=init_columns)  
    k += 1  
      
    # AccumulateComputeAverageValue  
    for j in range(1, cv_repeats + 1):  
        # Compatible R GenerateFileNameInFrontPlaceEmptyFormat  
        name = f' {j} {i[0]} r2_test.csv'  
        file_path = os.path.join(dire, name)  
          
        if not os.path.exists(file_path):  
            raise FileNotFoundError(f"NotFoundCorrectCorrespondingCrossverificationResultFile: {file_path}")  
              
        file = pd.read_csv(file_path)  
        file = file.iloc[:, 1:]  # Remove auto-increment sequenceColumnNumberColumn  
        r2_te_sum = r2_te_sum + file  
          
    r2_te_sum = r2_te_sum / cv_repeats  # ComputeaverageValue  
      
    # RebuildRowIndexAndColumnNameAsCorrectCorresponding Gamma and Cost ActualParametersValue  
    r2_te_sum.index = gamma_list  
    r2_te_sum.columns = cost_list  
      
    # --- OptimizeCheckFindLogic: Using stack + idxmax FastQuick lockFirstMaximumValue (gamma, cost) Coordinate ---
    stacked = r2_te_sum.stack()
    best_gamma, best_cost = stacked.idxmax()  # DefaultReturnsFirstMaximumValueCorrectCorresponding(row_index, col_name)TupleGroup
    fc_max_r2_te = stacked[(best_gamma, best_cost)]
    
    # RecordPerOneFeatureCombinationOptimalSuperParameters
    r2_te_list.append({  
        'features': i[0],   
        'best_hp_result': fc_max_r2_te,   
        'gamma': best_gamma,   
        'cost': best_cost  
    })  
  
# ColumnTableOneSingle-useConvertAs DataFrame
r2_te = pd.DataFrame(r2_te_list)

# EnsureColumnOrder
r2_te = r2_te[['features', 'best_hp_result', 'gamma', 'cost']]  
output_csv = output_svr_dir / 'r2_te.csv'  
r2_te.to_csv(output_csv, index=False)  
print(f"✅ Optimal SVR RowParametersExtractMappingTableAlreadySave: {output_csv}")