# gbdt_hp.py  
import os  
import pandas as pd  
from pathlib import Path  
  
  
# GetProject rootPathAndOutputPath  
CURRENT_FILE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", CURRENT_FILE_DIR.parent.parent))  
  
  
# S3 OutputTotalPath  
S3_OUTPUT_DIR = Path(os.environ.get("S3_OUTPUT_DIR", BASE_DIR / "output" / "S3-feature_selection"))  
  
  
# GBDT SuperParametersOutputSourceFileFolder (hp)  
dire = os.environ.get("GBDT_HP_OUTPUT_DIR", str(S3_OUTPUT_DIR / "gbdt" / "hp"))  
output_gbdt_dir = Path(os.environ.get("GBDT_OUTPUT_DIR", S3_OUTPUT_DIR / "gbdt"))  
  
  
# FeatureMappingTable  
feature_comb_file = S3_OUTPUT_DIR / "feature_combination.csv"  
importance_combination = pd.read_csv(feature_comb_file)  
  
  
# FromEnvironmentVariableGetSuperParametersGrid range  
shrinkage_str = os.environ.get("GBDT_SHRINKAGE_VALUES", "0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1")  
ntree_str = os.environ.get("GBDT_NTREE_VALUES", "100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000")  
  
  
shrinkage_list = [float(x) for x in shrinkage_str.split(",")]  
ntree_list = [int(x) for x in ntree_str.split(",")]  
  
  
cv_repeats = int(os.environ.get("GBDT_CV_REPEATS", 50))  
  
  
# UsingColumnTableCollect dict to replaceLowEffective DataFrame._append  
r2_te_list = []  
x = 1  
  
  
# EstablishParametersGrid occupancyPosition DataFrame IndexandColumnName  
init_index = list(range(len(shrinkage_list)))  
init_columns = [f"V{i}" for i in range(1, len(ntree_list) + 1)]  
  
  
print(f"StartAggregateAndSearchFindEachFeatureCombinationCorrectCorresponding GBDT OptimalTuningSuperParameters (Total {len(importance_combination)} Group)...")  
  
  
for i in importance_combination.values.tolist():  
    r2_te_sum = pd.DataFrame(0.0, index=init_index, columns=init_columns)  
    x += 1  
      
    # AccumulateComputeAverageValue  
    for j in range(1, cv_repeats + 1):  
        # Compatible R GenerateFileNameInFrontPlaceEmptyGrid format  
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
    r2_te_sum.index = shrinkage_list  
    r2_te_sum.columns = ntree_list  
      
    # --- OptimizeCheckFindLogic: Using stack + idxmax FastQuick lockFirstMaximumValue (shrinkage, ntree) Coordinate ---
    stacked = r2_te_sum.stack()
    best_shrinkage, best_ntree = stacked.idxmax()  # DefaultAutoFilterOutputFirstMaximumValue
    fc_max_r2_te = stacked[(best_shrinkage, best_ntree)]
      
    print(f'{x-1}.{i[0]}')  
    
    # RecordPerOneFeatureCombinationOptimalSuperParameters
    r2_te_list.append({  
        'features': i[0],   
        'best_hp_result': fc_max_r2_te,   
        'shrinkage': best_shrinkage,   
        'ntree': best_ntree  
    })  
  
  
# ColumnTableOneSingle-useConvertAs DataFrame
r2_te = pd.DataFrame(r2_te_list)

# EnsureOutput CSV ColumnWithHasDetermineOrder  
r2_te = r2_te[['features', 'best_hp_result', 'shrinkage', 'ntree']]  
output_csv = output_gbdt_dir / 'r2_te.csv'  
r2_te.to_csv(output_csv, index=False)  
print(f"✅ Optimal GBDT RowParametersExtractMappingTableAlreadySave: {output_csv}")