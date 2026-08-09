# ============================================================
# config.py
# run.py unifiedConfigurationFile
# AllModelParameters, Path, FilterFeatureCountAll modifications here
# ============================================================

from pathlib import Path

# ====================== ProjectPath ======================
BASE_DIR = Path(__file__).resolve().parent

# DATA_DIR = BASE_DIR / "data" # StoreS2-M1-importanceInput,ToFewNeedTo put data_after-pcc.csv or data.csv ItsInOfOne
DATA_DIR = "D:/MLY/ABO3/Machine learningML/Qf_G1/S1-pcc"
DATA_DIR = Path(DATA_DIR)

S2_CODE_DIR = BASE_DIR / "S2-M1-importance" # Store S2-M1-importance Code

# OUTPUT_DIR = BASE_DIR / "output" / "S2-M1-importance" # Store S2-M1-importance CodeProcessOutputFile
OUTPUT_DIR = "D:/MLY/ABO3/Machine learningML/Qf_G1/S2-M1-importance"
OUTPUT_DIR = Path(OUTPUT_DIR)
S3_DIR = BASE_DIR / "S3-feature_selection"
# S3_OUTPUT_DIR = BASE_DIR / "output" / "S3-feature_selection" # Store S3-feature_selection CodeProcessOutputFile
S3_OUTPUT_DIR = "D:/MLY/ABO3/Machine learningML/Qf_G1/S3-feature_selection"
S3_OUTPUT_DIR = Path(S3_OUTPUT_DIR)

# SVR / XGB / RF / GBDT (S3) PathPlan
SVR_CODE_DIR = S3_DIR / "svr"
SVR_OUTPUT_DIR = SVR_OUTPUT_DIR = S3_OUTPUT_DIR / "svr"
SVR_HP_OUTPUT_DIR = SVR_OUTPUT_DIR / "hp"

XGB_CODE_DIR = S3_DIR / "xgb"
XGB_OUTPUT_DIR = S3_OUTPUT_DIR / "xgb"
XGB_HP_OUTPUT_DIR = XGB_OUTPUT_DIR / "hp"

RF_S3_CODE_DIR = S3_DIR / "rf"
RF_S3_OUTPUT_DIR = S3_OUTPUT_DIR / "rf"
RF_S3_HP_OUTPUT_DIR = RF_S3_OUTPUT_DIR / "hp"

GBDT_S3_CODE_DIR = S3_DIR / "gbdt"
GBDT_S3_OUTPUT_DIR = S3_OUTPUT_DIR / "gbdt"
GBDT_S3_HP_OUTPUT_DIR = GBDT_S3_OUTPUT_DIR / "hp"

# Newly added: S4 PredictionStagePathPlan
S4_CODE_DIR = BASE_DIR / "S4-train_predict"
# S4_OUTPUT_DIR = BASE_DIR / "output" / "S4-train_predict"
S4_OUTPUT_DIR = "D:/MLY/ABO3/Machine learningML/Qf_G1/S4-train_predict"
S4_OUTPUT_DIR = Path(S4_OUTPUT_DIR)

# ====================== Rscript Path ======================
# Windows BottomSuggestionWrite Rscript.exe CompletePath
RSCRIPT_PATH = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"

# ====================== GeneralParameters ======================
TARGET = "targ"

# TrainingSetProportion(For example 0.7 TableShowTrainingSet 70%,TestSet 30%)
TRAIN_RATIO = 0.7

# FeatureImportance resamplingTotalTimes
# S2-M1-importanceIn gbdt/rf/xgb Importance insideExtractLoopTimes
N_RESAMPLING = 1000

# IsNoAndRowRunRow gbdt.R, rf.R, xgb.R
RUN_MODELS_IN_PARALLEL = True

# ====================== Newly added: AutoCheckpoint reuseandRun logicSkipControlSwitch ======================
SKIP_EXISTING_S2_OUTPUTS = True       # IsNoAutoSkipAlreadyViaHasOutputResult S2 Step
SKIP_EXISTING_S3_S4_OUTPUTS = True    # IsNoCorrect S3 (CV->HP->Optimal) and S4 Enable checkpoint detectionAvoidDuplicate redundant computation

# ====================== S3 ComputeMethodPipeline runRowEnableControlSwitch ======================
# 1 = Enable executionCorrectcorresponding full set CV -> HP -> Optimal Pipeline,0 = NotRunRow.DefaultAllAs 1
RUN_SVR_S3 = 1
RUN_XGB_S3 = 1
RUN_RF_S3 = 1
RUN_GBDT_S3 = 1

# ====================== S4 PerformanceSimulatePredictionResamplingTimes ======================
S4_REPEATS = 1000

# ====================== R InsidePartCoreCountControl ======================
GBDT_N_CORES = 1 # S2-M1-importanceIn gbdtControlAndRown_core 
RF_N_CORES = 10 # S2-M1-importanceIn rfControlAndRown_core 
XGB_N_CORES = 10 # S2-M1-importanceIn xgbControlAndRown_core

# ====================== GBDT Parameters(S2-M1-importance) ======================
GBDT_SHRINKAGE = 0.01
GBDT_N_TREES = 3000
GBDT_INTERACTION_DEPTH = 3
GBDT_BAG_FRACTION = 0.5
GBDT_TRAIN_FRACTION = 0.8
GBDT_MINOBSINNODE = 3
GBDT_CV_FOLDS = 10

# ====================== RF Parameters(S2-M1-importance) ======================
RF_TUNING_REPEATS = 30
RF_MAXNODES_VALUES = list(range(4, 61, 4))
RF_NTREE_VALUES = list(range(300, 3001, 300))

# ====================== XGB Parameters(S2-M1-importance) ======================
XGB_TUNING_REPEATS = 30
XGB_MAX_DEPTH_VALUES = [2, 4, 6, 8, 10]
XGB_LEARNING_RATE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
XGB_NROUND_FIXED = 25

# ====================== FeatureFilterParameters ======================
N_TOP_FEATURES = 7

# ====================== SVR ModelParameters(S3-feature_selection/svr) ======================
SVR_N_CORES = 10                  # SVR AndRowCalculateUsingCoreCount (svr_cv.R and svr_optimal.R UnifiedOneUsingThis core count)
SVR_CV_FOLDS = 10                 # K-fold crossverification
SVR_CV_REPEATS = 50               # Repeat how many roundsRandomSplit intoEvaluation
SVR_OPTIMAL_REPEATS = 1000        # OptimalSuperParametersBottomResamplingEvaluationIterationSamplingTimes

# Gamma and Cost Optimization grid range
SVR_GAMMA_VALUES = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
SVR_COST_VALUES = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# ====================== XGB ModelParameters(S3-feature_selection/xgb) ======================
XGB_S3_N_CORES = 12               # XGB Parallel core count
XGB_S3_CV_FOLDS = 10              # Fold count
XGB_S3_CV_REPEATS = 50            # cv RepeatTimes
XGB_S3_OPTIMAL_REPEATS = 1000     # ResamplingSampleaverageTimes
XGB_S3_MAX_DEPTH_VALUES = [2, 4, 6, 8, 10]
XGB_S3_LEARNING_RATE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
XGB_S3_NROUND_OPTIMAL = 25

# ====================== RF ModelParameters(S3-feature_selection/rf) ======================
RF_S3_N_CORES = 15                 # RF Parallel core count
RF_S3_CV_FOLDS = 10                # Fold count
RF_S3_CV_REPEATS = 50              # cv RepeatTimes
RF_S3_OPTIMAL_REPEATS = 1000       # ResamplingSampleaverageTimes
RF_S3_MAXNODES_VALUES = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
RF_S3_NTREE_VALUES = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]

# ====================== GBDT ModelParameters(S3-feature_selection/gbdt) ======================
GBDT_S3_N_CORES = 10               # GBDT Parallel core count
GBDT_S3_CV_FOLDS = 10              # Fold count
GBDT_S3_CV_REPEATS = 50            # cv RepeatTimes
GBDT_S3_OPTIMAL_REPEATS = 1000     # ResamplingSampleaverageTimes

# GBDT Grid range
GBDT_S3_SHRINKAGE_VALUES = [0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.06, 0.08]
GBDT_S3_NTREE_VALUES = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]