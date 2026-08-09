# rf_hp.py  
import os  
import pandas as pd  
from pathlib import Path  
  
  
# GetProject rootPathAndOutputPath  
CURRENT_FILE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", CURRENT_FILE_DIR.parent.parent))  
  
  
# S3 OutputTotalPath  
S3_OUTPUT_DIR = Path(os.environ.get("S3_OUTPUT_DIR", BASE_DIR / "output" / "S3-feature_selection"))  
  
  
# RF SuperParametersOutputSourceFileFolder (hp)  
dire = os.environ.get("RF_HP_OUTPUT_DIR", str(S3_OUTPUT_DIR / "rf" / "hp"))  
output_rf_dir = Path(os.environ.get("RF_OUTPUT_DIR", S3_OUTPUT_DIR / "rf"))  
  
  
# FeatureMappingTable  
feature_comb_file = S3_OUTPUT_DIR / "feature_combination.csv"  
importance_combination = pd.read_csv(feature_comb_file)  
  
  
# FromEnvironmentVariableGetSuperParametersRange  
maxnodes_str = os.environ.get("RF_MAXNODES_VALUES", "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60")  
ntree_str = os.environ.get("RF_NTREE_VALUES", "200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000")  
  
  
maxnodes_list = [int(x) for x in maxnodes_str.split(",")]  
ntree_list = [int(x) for x in ntree_str.split(",")]  
  
  
cv_repeats = int(os.environ.get("RF_CV_REPEATS", 50))  
  
  
# UsingColumnTableCollect dict to replaceLowEffective DataFrame._append  
r2_te_list = []  
x = 1  
  
  
# Based onParametersColumnTableDynamicConfiguration Index and Columns OccupyPositionSymbol  
init_index = list(range(len(maxnodes_list)))  
init_columns = [f"V{i}" for i in range(1, len(ntree_list) + 1)]  
  
  
print(f"StartAggregateAndSearchFindEachFeatureCombinationCorrectCorresponding Random Forest OptimalTuningSuperParameters (Total {len(importance_combination)} Group)...")  
  
  
for i in importance_combination.values.tolist():  
    r2_te_sum = pd.DataFrame(0.0, index=init_index, columns=init_columns)  
    x = x + 1  
      
    # AccumulateComputeAverageValue  
    for j in range(1, cv_repeats + 1):  
        # Compatible R GenerateFileNameInFrontPlaceEmptyFormat  
        name = f' {j} {i[0]} r2_test.csv'  
        file_path = os.path.join(dire, name)  
          
        if not os.path.exists(file_path):  
            raise FileNotFoundError(f"NotFoundCorrectCorrespondingCrossverificationResultFile: {file_path}")  
              
        file = pd.read_csv(file_path)  
          
        if file.shape[1] > len(ntree_list):  
            file = file.iloc[:, 1:]  
              
        r2_te_sum = r2_te_sum + file  
          
    r2_te_sum = r2_te_sum / cv_repeats  # ComputeaverageValue  
      
    # RebuildRowIndexAndColumnNameAsCorrectCorrespondingParametersValue  
    r2_te_sum.index = maxnodes_list  
    r2_te_sum.columns = ntree_list  
      
    # --- OptimizeCheckFindLogic: Using stack + idxmax FastQuick lockFirstMaximumValue (maxnodes, ntree) Coordinate ---
    stacked = r2_te_sum.stack()
    best_maxnodes, best_ntree = stacked.idxmax()  # DefaultAutoFilterOutputFirstValueMaximumCoordinate
    fc_max_r2_te = stacked[(best_maxnodes, best_ntree)]
      
    print(f'{x-1}.{i[0]}')  
    
    # RecordPerOneFeatureCombinationOptimalSuperParameters
    r2_te_list.append({  
        'features': i[0],   
        'best_hp_result': fc_max_r2_te,   
        'maxnodes': best_maxnodes,   
        'ntree': best_ntree  
    })  
  
  
# ColumnTableOneSingle-useConvertAs DataFrame
r2_te = pd.DataFrame(r2_te_list)

# EnsureOutput CSV ColumnWithHasDetermineOrder  
r2_te = r2_te[['features', 'best_hp_result', 'maxnodes', 'ntree']]  
output_csv = output_rf_dir / 'r2_te.csv'  
r2_te.to_csv(output_csv, index=False)  
print(f"✅ Optimal RF RowParametersExtractMappingTableAlreadySave: {output_csv}")