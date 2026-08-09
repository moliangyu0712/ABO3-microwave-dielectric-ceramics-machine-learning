# run.py
# ============================================================
# OneKey runRow:
# 1. AndRoworStringRowRunRow gbdt.R, rf.R, xgb.R (Checkpoint reuse detectionAlreadySupport)
# 2. ThreeModelAllCompletedBackRunRow range.py Generate importantFeature
# 3. RunRow Exhaustive method.py GenerateFeatureCombinationSubsetdata
# 4. **AndRowand S3/S4 AllChainPath checkpointSkipMechanism**:
#    Based on config.py In RUN_XXX_S3 Parameters,DetermineIsNoActivateBelowPipeline: 
#    - Pipeline A: SVR  (S3 CV/HP/Optimal -> S4 SVR Chain)
#    - Pipeline B: XGB  (S3 CV/HP/Optimal -> S4 XGB Chain)
#    - Pipeline C: RF   (S3 CV/HP/Optimal -> S4 RF Chain)
#    - Pipeline D: GBDT (S3 CV/HP/Optimal -> S4 GBDT Chain)
#    AllChainIntelligent path identification: EachCalculateProductExistsWhenAutoCrossStepSkipPrevent duplicate run.
# ============================================================

import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import config

def list_to_env_string(values):
    return ",".join(str(v) for v in values)

def check_s2_outputs_exist(script_path):
    """
    Check S2 EachModelCorrectCorrespondingOutput CSV FileIsNoAlreadyViaExists
    """
    output_dir = config.OUTPUT_DIR
    stem = script_path.stem.lower()

    if stem == "gbdt":
        files_to_check = [output_dir / "gbdt_im1.csv", output_dir / "gbdt_im2.csv"]
    elif stem == "rf":
        files_to_check = [output_dir / "rf_im1.csv", output_dir / "rf_im2.csv"]
    elif stem == "xgb":
        files_to_check = [output_dir / "xgb_im1.csv"]
    else:
        return False

    return all(f.exists() for f in files_to_check)

def check_s3_cv_files_strictly(model_name, s3_hp_dir, cv_repeats):
    """
    Strict scanEachFeatureFormulaandPerRepeatCorrectCorresponding R2/RMSE OutputFileIsNoComplete.
    """
    feat_comb_file = config.S3_OUTPUT_DIR / "feature_combination.csv"
    if not feat_comb_file.exists():
        return False
    
    try:
        feat_df = pd.read_csv(feat_comb_file)
        formulas = feat_df.iloc[:, 0].tolist()
    except Exception:
        return False

    suffix_list = ["r2_test.csv", "r2_train.csv", "rmse_test.csv", "rmse_train.csv"]
    s3_hp_path = Path(s3_hp_dir)
    
    # AllQuantityScan judgmentIsNoHasFileLost
    for formula in formulas:
        for r in range(1, cv_repeats + 1):
            for suffix in suffix_list:
                # Compatible R paste DefaultGenerate " tt name suffix" WithEmptyWhitespace first charFileNameName
                filename = f" {r} {formula} {suffix}"
                file_path = s3_hp_path / filename
                if not file_path.exists():
                    return False
    return True

def build_env():
    """
    BuildPass toAll R Scriptand Python Script system environmentVariable
    """
    env = os.environ.copy()

    # Force Python SubprocessCorrectStandardFlowUsing UTF-8 Encoding,Prevent Windows ControlPlatform reportWrong
    env["PYTHONIOENCODING"] = "utf-8"

    # Main projectPath
    env["PROJECT_ROOT"] = str(config.BASE_DIR)
    env["DATA_DIR"] = str(config.DATA_DIR)
    env["OUTPUT_DIR"] = str(config.OUTPUT_DIR)
    env["S3_DIR"] = str(config.S3_DIR)
    env["S3_OUTPUT_DIR"] = str(config.S3_OUTPUT_DIR)
    
    # S4 PathAndParameters
    env["S4_CODE_DIR"] = str(config.S4_CODE_DIR)
    env["S4_OUTPUT_DIR"] = str(config.S4_OUTPUT_DIR)
    env["S4_REPEATS"] = str(config.S4_REPEATS)

    # TargetField and partitioning
    env["TARGET"] = str(config.TARGET)
    env["TRAIN_RATIO"] = str(config.TRAIN_RATIO)
    env["N_RESAMPLING"] = str(config.N_RESAMPLING)

    # ThreeLargeMainFeatureModelCoreCountControl
    env["GBDT_N_CORES"] = str(config.GBDT_N_CORES)
    env["RF_N_CORES"] = str(config.RF_N_CORES)
    env["XGB_NTHREAD"] = str(config.XGB_N_CORES)

    env["GBDT_SHRINKAGE"] = str(config.GBDT_SHRINKAGE)
    env["GBDT_N_TREES"] = str(config.GBDT_N_TREES)
    env["GBDT_INTERACTION_DEPTH"] = str(config.GBDT_INTERACTION_DEPTH)
    env["GBDT_BAG_FRACTION"] = str(config.GBDT_BAG_FRACTION)
    env["GBDT_TRAIN_FRACTION"] = str(config.GBDT_TRAIN_FRACTION)
    env["GBDT_MINOBSINNODE"] = str(config.GBDT_MINOBSINNODE)
    env["GBDT_CV_FOLDS"] = str(config.GBDT_CV_FOLDS)

    env["RF_TUNING_REPEATS"] = str(config.RF_TUNING_REPEATS)
    env["RF_MAXNODES_VALUES"] = list_to_env_string(config.RF_MAXNODES_VALUES)
    env["RF_NTREE_VALUES"] = list_to_env_string(config.RF_NTREE_VALUES)

    env["XGB_TUNING_REPEATS"] = str(config.XGB_TUNING_REPEATS)
    env["XGB_MAX_DEPTH_VALUES"] = list_to_env_string(config.XGB_MAX_DEPTH_VALUES)
    env["XGB_LEARNING_RATE_VALUES"] = list_to_env_string(config.XGB_LEARNING_RATE_VALUES)
    env["XGB_NROUND_FIXED"] = str(config.XGB_NROUND_FIXED)

    env["TOP_N_FEATURES"] = str(config.N_TOP_FEATURES)

    # A: SVR Parameters
    env["SVR_CODE_DIR"] = str(config.SVR_CODE_DIR)
    env["SVR_OUTPUT_DIR"] = str(config.SVR_OUTPUT_DIR)
    env["SVR_HP_OUTPUT_DIR"] = str(config.SVR_HP_OUTPUT_DIR)
    env["SVR_N_CORES"] = str(config.SVR_N_CORES)
    env["SVR_CV_FOLDS"] = str(config.SVR_CV_FOLDS)
    env["SVR_CV_REPEATS"] = str(config.SVR_CV_REPEATS)
    env["SVR_OPTIMAL_REPEATS"] = str(config.SVR_OPTIMAL_REPEATS)
    env["SVR_GAMMA_VALUES"] = list_to_env_string(config.SVR_GAMMA_VALUES)
    env["SVR_COST_VALUES"] = list_to_env_string(config.SVR_COST_VALUES)

    # B: XGB Parameters
    env["XGB_CODE_DIR"] = str(config.XGB_CODE_DIR)
    env["XGB_OUTPUT_DIR"] = str(config.XGB_OUTPUT_DIR)
    env["XGB_HP_OUTPUT_DIR"] = str(config.XGB_HP_OUTPUT_DIR)
    env["FEATURE_COMB_DIR"] = str(config.S3_OUTPUT_DIR / "feature_combination")
    env["FEATURE_COMB_FILE"] = str(config.S3_OUTPUT_DIR / "feature_combination.csv")
    env["XGB_PARA_FILE"] = str(config.XGB_OUTPUT_DIR / "r2_te.csv")
    env["XGB_N_CORES"] = str(config.XGB_S3_N_CORES)
    env["XGB_CV_FOLDS"] = str(config.XGB_S3_CV_FOLDS)
    env["XGB_CV_REPEATS"] = str(config.XGB_S3_CV_REPEATS)
    env["XGB_OPTIMAL_REPEATS"] = str(config.XGB_S3_OPTIMAL_REPEATS)
    env["XGB_MAX_DEPTH_VALUES"] = list_to_env_string(config.XGB_S3_MAX_DEPTH_VALUES)
    env["XGB_LEARNING_RATE_VALUES"] = list_to_env_string(config.XGB_S3_LEARNING_RATE_VALUES)
    env["XGB_NROUND_OPTIMAL"] = str(config.XGB_S3_NROUND_OPTIMAL)

    # C: RF Parameters
    env["RF_CODE_DIR"] = str(config.RF_S3_CODE_DIR)
    env["RF_OUTPUT_DIR"] = str(config.RF_S3_OUTPUT_DIR)
    env["RF_HP_OUTPUT_DIR"] = str(config.RF_S3_HP_OUTPUT_DIR)
    env["RF_N_CORES"] = str(config.RF_S3_N_CORES)
    env["RF_CV_FOLDS"] = str(config.RF_S3_CV_FOLDS)
    env["RF_CV_REPEATS"] = str(config.RF_S3_CV_REPEATS)
    env["RF_OPTIMAL_REPEATS"] = str(config.RF_S3_OPTIMAL_REPEATS)
    env["RF_MAXNODES_VALUES"] = list_to_env_string(config.RF_S3_MAXNODES_VALUES)
    env["RF_NTREE_VALUES"] = list_to_env_string(config.RF_S3_NTREE_VALUES)

    # D: GBDT Parameters
    env["GBDT_CODE_DIR"] = str(config.GBDT_S3_CODE_DIR)
    env["GBDT_OUTPUT_DIR"] = str(config.GBDT_S3_OUTPUT_DIR)
    env["GBDT_HP_OUTPUT_DIR"] = str(config.GBDT_S3_HP_OUTPUT_DIR)
    env["GBDT_N_CORES"] = str(config.GBDT_S3_N_CORES)
    env["GBDT_CV_FOLDS"] = str(config.GBDT_S3_CV_FOLDS)
    env["GBDT_CV_REPEATS"] = str(config.GBDT_S3_CV_REPEATS)
    env["GBDT_OPTIMAL_REPEATS"] = str(config.GBDT_S3_OPTIMAL_REPEATS)
    env["GBDT_SHRINKAGE_VALUES"] = list_to_env_string(config.GBDT_S3_SHRINKAGE_VALUES)
    env["GBDT_NTREE_VALUES"] = list_to_env_string(config.GBDT_S3_NTREE_VALUES)

    return env

def ensure_dirs():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.S3_DIR.mkdir(parents=True, exist_ok=True)
    config.S3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    config.SVR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SVR_HP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.XGB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.XGB_HP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RF_S3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RF_S3_HP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.GBDT_S3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.GBDT_S3_HP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    config.S4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def check_input_data():
    qf_after_pcc = config.DATA_DIR / "data_after-pcc.csv"
    qf = config.DATA_DIR / "data.csv"
    if not qf_after_pcc.exists() and not qf.exists():
        raise FileNotFoundError("Missing basisInputdata.")

def run_r_script(script_path, env):
    script_path = Path(script_path)
    rscript_path = Path(config.RSCRIPT_PATH)
    if not rscript_path.exists():
         raise FileNotFoundError(f"NotFound Rscript.exe: {rscript_path}")

    cmd = [str(rscript_path), str(script_path)]
    print(f"\n🚀 Start R Script: {script_path.name}")
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    
    log_path = config.OUTPUT_DIR / f"{script_path.stem}.log"
    err_path = config.OUTPUT_DIR / f"{script_path.stem}.error.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    err_path.write_text(process.stderr, encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(f"R Script {script_path.name} RunRowFailed")
    print(f"✅ R ScriptCompleted: {script_path.name}")
    return script_path.name

def run_python_script(script_path, env, args=None):
    script_path = Path(script_path)
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend([str(x) for x in args])

    print(f"\n🚀 Start Python Script: {script_path.name}")
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    
    log_name = script_path.stem.replace(" ", "_")
    log_path = config.OUTPUT_DIR / f"{log_name}.log"
    err_path = config.OUTPUT_DIR / f"{log_name}.error.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    err_path.write_text(process.stderr, encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(f"Python {script_path.name} RunRowFailed")
    print(f"✅ Python ScriptCompleted: {script_path.name}")
    return script_path.name

# ==================== FeatureFilterBackComputeMethodSubTaskPipeline wrapper ====================

def run_s4_pipeline(model_name, env):
    """
    General S4 Full workflowAutoExecuteRow (SupportInsidePartFormulaFileFolderLevel checkpoint identification)
    """
    print(f"\n==================== Start {model_name.upper()} S4 ModelingPredictionPipeline ====================")
    
    # 1. RunRow sorted_result.py Create5 R2 MostHighResultFileFolder
    sorted_script = config.S4_CODE_DIR / "sorted_result.py"
    run_python_script(sorted_script, env, args=["--model", model_name])
    
    # 2. ReadGenerate top_5_formulas.csv
    top5_csv_path = config.S4_OUTPUT_DIR / model_name / "top_5_formulas.csv"
    if not top5_csv_path.exists():
        raise FileNotFoundError(f"NotDetectTo {model_name} Top-5 dataMappingTable.")
        
    top5_df = pd.read_csv(top5_csv_path)
    
    if model_name == "svr":
        hp1_col, hp2_col = "gamma", "cost"
        r_script_path = config.S4_CODE_DIR / "svr.R"
    elif model_name == "xgb":
        hp1_col, hp2_col = "max_depth", "eta"
        r_script_path = config.S4_CODE_DIR / "xgb.R"
    elif model_name == "rf":
        hp1_col, hp2_col = "maxnodes", "ntree"
        r_script_path = config.S4_CODE_DIR / "rf.R"
    elif model_name == "gbdt":
        hp1_col, hp2_col = "shrinkage", "n.trees"
        r_script_path = config.S4_CODE_DIR / "gbdt.R"
    else:
        raise ValueError(f"NotReceiveSupportComputeMethodType: {model_name}")

    crop_script = config.S4_CODE_DIR / "crop_dataset.py"
    process_script = config.S4_CODE_DIR / "result_processing.py"
    skip_s3_s4 = getattr(config, "SKIP_EXISTING_S3_S4_OUTPUTS", True)
    
    # 3. Correct 5 FormulaProceedIteration
    for idx, row in top5_df.iterrows():
        formula_str = str(row["formula"])
        hp1_val = row[hp1_col]
        hp2_val = row[hp2_col]
        
        formula_dir = config.S4_OUTPUT_DIR / model_name / formula_str
        result_csv = formula_dir / "result_data.csv"
        
        # LocalPartCheckpoint determination
        if skip_s3_s4 and result_csv.exists():
            print(f"   ℹ️ [{model_name.upper()} S4 ({idx+1}/5)] DetectToFeatureFit productAlreadyExists.SkipThisFormulaRunRow: {formula_str}")
            continue

        formula_env = env.copy()
        formula_env["S4_FORMULA"] = formula_str
        formula_env["S4_MODEL_DIR"] = str(formula_dir)
        formula_env["S4_INPUT_CSV"] = str(formula_dir / f"{formula_str}.csv")
        formula_env["S4_HP_P1"] = str(hp1_val)
        formula_env["S4_HP_P2"] = str(hp2_val)

        print(f"\n👉 [{model_name.upper()} S4 ({idx+1}/5)] CurrentProcessingFeatureMode: {formula_str}")
        run_python_script(crop_script, env=formula_env)
        run_r_script(r_script_path, env=formula_env)
        run_python_script(process_script, env=formula_env)

    # 4. Correct 5 SubFormulaOverallEvaluationAndCompleted overall_evaluation
    eval_script = config.S4_CODE_DIR / "R2-RMSE-MAE.py"
    eval_env = env.copy()
    eval_env["S4_MODEL_TYPE_DIR"] = str(config.S4_OUTPUT_DIR / model_name)
    run_python_script(eval_script, env=eval_env)
    
    print(f"\n==================== ✅ {model_name.upper()} S4 Pipeline runRowEnd!====================\n")

# ============== UnifiedOneCheckpoint decisionExecuteRowPipeline (S3 RunRowEndBack,Defer immediatelyExecuteRow S4 Pipeline) ==============

def run_model_pipeline(model_name, env):
    """
    GeneralFeatureSelect (S3) AndPrediction (S4) Checkpoint protectionAndLaunch pipeline
    """
    if getattr(config, f"RUN_{model_name.upper()}_S3", 1) != 1:
        return

    if model_name == "svr":
        s3_out_dir = config.SVR_OUTPUT_DIR
        s3_hp_dir = config.SVR_HP_OUTPUT_DIR
        cv_script = config.SVR_CODE_DIR / "svr_cv.R"
        hp_script = config.SVR_CODE_DIR / "svr_hp.py"
        optimal_script = config.SVR_CODE_DIR / "svr_optimal.R"
        cv_repeats = int(config.SVR_CV_REPEATS)
    elif model_name == "xgb":
        s3_out_dir = config.XGB_OUTPUT_DIR
        s3_hp_dir = config.XGB_HP_OUTPUT_DIR
        cv_script = config.XGB_CODE_DIR / "xgb_cv.R"
        hp_script = config.XGB_CODE_DIR / "xgb_hp.py"
        optimal_script = config.XGB_CODE_DIR / "xgb_optimal.R"
        cv_repeats = int(config.XGB_S3_CV_REPEATS)
    elif model_name == "rf":
        s3_out_dir = config.RF_S3_OUTPUT_DIR
        s3_hp_dir = config.RF_S3_HP_OUTPUT_DIR
        cv_script = config.RF_S3_CODE_DIR / "rf_cv.R"
        hp_script = config.RF_S3_CODE_DIR / "rf_hp.py"
        optimal_script = config.RF_S3_CODE_DIR / "rf_optimal.R"
        cv_repeats = int(config.RF_S3_CV_REPEATS)
    elif model_name == "gbdt":
        s3_out_dir = config.GBDT_S3_OUTPUT_DIR
        s3_hp_dir = config.GBDT_S3_HP_OUTPUT_DIR
        cv_script = config.GBDT_S3_CODE_DIR / "gbdt_cv.R"
        hp_script = config.GBDT_S3_CODE_DIR / "gbdt_hp.py"
        optimal_script = config.GBDT_S3_CODE_DIR / "gbdt_optimal.R"
        cv_repeats = int(config.GBDT_S3_CV_REPEATS)
    else:
        return

    optimal_file = s3_out_dir / "final_sample_average_metrics.csv"
    hp_file = s3_out_dir / "r2_te.csv"
    s4_eval_file = config.S4_OUTPUT_DIR / model_name / "overall_evaluation.csv"

    # ====================== Checkpoint readiness check ======================
    skip_enabled = getattr(config, "SKIP_EXISTING_S3_S4_OUTPUTS", True)
    
    cv_complete = False
    if hp_file.exists(): 
        # IfSuperParameter optimizationTable r2_te.csv AlreadyViaSuccessExport,Proof CV StagedataIsCompleteRun completeReceiveGood
        cv_complete = True
    elif skip_enabled:
        # cv NotFully completeWhen,Using strict MethodCorrectEachCombinationGrid 4  CSV FileCompleteQuantityStrict judgment
        cv_complete = check_s3_cv_files_strictly(model_name, s3_hp_dir, cv_repeats)

    hp_complete = hp_file.exists()
    optimal_complete = optimal_file.exists()
    s4_complete = s4_eval_file.exists()

    # AllChainPathAlreadyCompletion check
    if skip_enabled and cv_complete and hp_complete and optimal_complete and s4_complete:
        print(f"ℹ️ DetectTo {model_name.upper()} AlreadyHasComplete S3 & S4 RepeatCorrectRatioIntoResultTable.SafetySkipThisModelChain.")
        return

    print(f"\n==================== [Checkpoint detection: {model_name.upper()} ExecuteRowFlowAnalysis] ====================")

    # 1. RunRow S3 CV Grid search (R ScriptInsideWillFilterExistsFileImplement checkpoint resume)
    if skip_enabled and cv_complete:
        print(f"ℹ️ [{model_name.upper()} S3 CV] AlreadyHasCompleteGridCalculatedataSet.Skip CV Grid computation.")
    else:
        print(f"🚀 [{model_name.upper()} S3 CV] AndRowsupplementorRun complete set CV Grid search optimizationIn...")
        run_r_script(cv_script, env)

    # 2. RunRow S3 HP Hyperparameter aggregation extraction
    if skip_enabled and hp_complete:
        print(f"ℹ️ [{model_name.upper()} S3 HP] OptimalSuperParametersMappingFile r2_te.csv AlreadyGenerate.Skip HP RunRow.")
    else:
        print(f"🚀 [{model_name.upper()} S3 HP] HeavyNewAggregateOptimalSuperParameter mapping configCorrectTable...")
        run_python_script(hp_script, env)

    # 3. RunRow S3 Optimal Monte Carloverification
    if skip_enabled and optimal_complete:
        print(f"ℹ️ [{model_name.upper()} S3 Optimal] TestFit final EvaluationMetricExists.Skip Optimal.")
    else:
        print(f"🚀 [{model_name.upper()} S3 Optimal] OverallCalculateFinalMonte CarloTableCurrentaverageValueEvaluationIn...")
        run_r_script(optimal_script, env)

    # 4. Deferred trigger S4 ActualTestEvaluationCalculate
    if skip_enabled and s4_complete:
        print(f"ℹ️ [{model_name.upper()} S4 predict] Final S4 ColumnTrim fit and overall_evaluation etc.MetricAlreadyAllExists.")
    else:
        print(f"🚀 [{model_name.upper()} S4 predict] ForCorrect Top-5 R2 OptimalSelectCombination,EnableSamplePredictionRatioCorrectStatisticsQuantityEvaluation...")
        run_s4_pipeline(model_name, env)

# Encapsulated multi-threaded scheduling call
def run_svr_pipeline(env):  run_model_pipeline("svr", env)
def run_xgb_pipeline(env):  run_model_pipeline("xgb", env)
def run_rf_pipeline(env):   run_model_pipeline("rf", env)
def run_gbdt_pipeline(env): run_model_pipeline("gbdt", env)

def main():
    print("==================================================")
    print("MACHINE_LEARNING Multi-stageOneKey joint pipelineStart")
    print("==================================================")

    ensure_dirs()
    check_input_data()
    env = build_env()

    gbdt_script = config.S2_CODE_DIR / "gbdt.R"
    rf_script = config.S2_CODE_DIR / "rf.R"
    xgb_script = config.S2_CODE_DIR / "xgb.R"
    range_script = config.S2_CODE_DIR / "range.py"
    exhaustive_script = config.S3_DIR / "Exhaustive method.py"

    r_scripts = [gbdt_script, rf_script, xgb_script]

    # DetectAndFilterNotCompleted S2 GBDT/RF/XGB Script
    skip_s2_enabled = getattr(config, "SKIP_EXISTING_S2_OUTPUTS", True)
    scripts_to_run = []
    
    for script in r_scripts:
        if skip_s2_enabled and check_s2_outputs_exist(script):
            print(f"ℹ️ DetectTo {script.name} AlreadyWithHasCompleteOutputResult.SkipThisModelHeavyNewBuild.")
        else:
            scripts_to_run.append(script)

    if scripts_to_run:
        if config.RUN_MODELS_IN_PARALLEL:
            workers = min(3, len(scripts_to_run))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_script = {executor.submit(run_r_script, s, env): s for s in scripts_to_run}
                for future in as_completed(future_to_script):
                    future.result()
        else:
            for script in scripts_to_run:
                run_r_script(script, env)
    else:
        print("💡 All S2 FeatureImportanceWeightEvaluation scriptOutputFileAverageAlreadyInLocalExists,AlreadyAllSkip!")

    # FeatureRelatedPerformance ratioCorrectandScheduleColumnExtractGetFront TOP Feature
    run_python_script(script_path=range_script, env=env, args=[config.N_TOP_FEATURES])
    
    # GenerateFormulaCombination CSV Exhaustive subset
    run_python_script(script_path=exhaustive_script, env=env)

    # 4. AndRowScheduleFourConditional computationMethodAllChainPathTask
    active_pipelines = []
    if getattr(config, "RUN_SVR_S3", 1) == 1:
        active_pipelines.append(("SVR Pipeline", run_svr_pipeline))
    if getattr(config, "RUN_XGB_S3", 1) == 1:
        active_pipelines.append(("XGB Pipeline", run_xgb_pipeline))
    if getattr(config, "RUN_RF_S3", 1) == 1:
        active_pipelines.append(("RF Pipeline", run_rf_pipeline))
    if getattr(config, "RUN_GBDT_S3", 1) == 1:
        active_pipelines.append(("GBDT Pipeline", run_gbdt_pipeline))

    if active_pipelines:
        with ThreadPoolExecutor(max_workers=len(active_pipelines)) as pipeline_executor:
            future_to_name = {pipeline_executor.submit(func, env): name for name, func in active_pipelines}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                future.result()
                print(f"🏁 AndRowPipelineMainTaskAlreadyFull completion: {name}")

    print("\n==================================================")
    print("🎉 S3FeatureFilterandS4PerformanceSimulateLargePipelineOneKeyCalculateAllCircleFullEnd!")
    print("==================================================")

if __name__ == "__main__":
    main()