import os
import subprocess
import shutil
from pathlib import Path

# ===================== [CorePathConfiguration,Based onActualModify] =====================
# 1. CodeRootDirectory: config.py / run.py / Thisbatch_run.py LocationInDirectory
CODE_ROOT = r"D:\MLY\ABO3\code\machine_learning"

# 2. dataTotalRootDirectory: AllBusiness sub-FileFolderAllInHere(OutputAlso unifyOneExistsHereSubFileFolderInside)
DATA_ROOT = r"D:\MLY\ABO3\Machine learningML"

# 3. WhitelistCSVFilePath(SingleColumn,PerRowOneNeedTo runRowSubFileFolderName)
WHITELIST_CSV = r"D:\MLY\ABO3\Machine learningML\run_list.csv"

# 4. FixedFileName & Nested sub-Directory
DATA_SUB_FOLDER = "S1-pcc"   # EachBusiness sub-FileFolderBottomdataInsideLayerDirectory
DATA_FILE = "data_after-pcc.csv"

# 5. ConfigurationFile, RunRowScript, BackupFile
CONFIG_FILE = Path(CODE_ROOT) / "config.py"
RUN_SCRIPT = Path(CODE_ROOT) / "run.py"
CONFIG_BACKUP = Path(CODE_ROOT) / "config_backup.py"
# ======================================================================


def backup_original_config():
    """First runRowBackup original config.py,OnlyBackupOneTime"""
    if not CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
        print(f"✅ AlreadyBackup originalConfigurationFileTo: {CONFIG_BACKUP}\n")


def restore_original_config():
    """RunRowAllTaskBackRestoreOriginalConfiguration"""
    if CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        print("🔄 AlreadyRestoreAsOriginal config.py Configuration\n")


def modify_config(top_folder: Path):
    """
    CorrectRuleThen: 
    1. DATA_DIR = OutsideLayerFileFolder/S1-pcc (Ensure readablecsv,FixFileMissing reportWrong)
    2. S2/S3/S4 OutputDirectory = OutsideLayer businessFileFolderBottom(anddataPlaceInOneFrom,Restore your original habits)
    3. FullPreserve Path() LanguageMethod,Avoid / ConcatenateReportWrong
    """
    folder_name = top_folder.name
    # dataTrueActualPath: OutsideLayerFileFolder + S1-pcc
    real_data_dir = top_folder / DATA_SUB_FOLDER

    # ========== KeyModification: OutputDirectory Built at[dataSubFileFolder]InsidePart,Rather thanCodeDirectory ==========
    s2_out = top_folder / f"S2-M1-importance_{folder_name}"
    s3_out = top_folder / f"S3-feature_selection_{folder_name}"
    s4_out = top_folder / f"S4-train_predict_{folder_name}"

    # ReadOriginalConfigurationInsideContent
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    new_lines = []
    for line in lines:
        strip_line = line.strip()
        # dataDirectory: Point to S1-pcc
        if strip_line.startswith("DATA_DIR = "):
            path_str = str(real_data_dir).replace(os.sep, "/")
            new_line = f'DATA_DIR = Path("{path_str}")'
            new_lines.append(new_line)
        # S2 Output: dataOutsideLayerFileFolderBottom
        elif strip_line.startswith("OUTPUT_DIR = "):
            path_str = str(s2_out).replace(os.sep, "/")
            new_line = f'OUTPUT_DIR = Path("{path_str}")'
            new_lines.append(new_line)
        # S3 Output: dataOutsideLayerFileFolderBottom
        elif strip_line.startswith("S3_OUTPUT_DIR = "):
            path_str = str(s3_out).replace(os.sep, "/")
            new_line = f'S3_OUTPUT_DIR = Path("{path_str}")'
            new_lines.append(new_line)
        # S4 Output: dataOutsideLayerFileFolderBottom
        elif strip_line.startswith("S4_OUTPUT_DIR = "):
            path_str = str(s4_out).replace(os.sep, "/")
            new_line = f'S4_OUTPUT_DIR = Path("{path_str}")'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # WriteReturnConfigurationFile
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    print(f"📝 ConfigurationUpdateCompleted:")
    print(f"   dataDirectory(DATA_DIR): {real_data_dir}")
    print(f"   S2OutputDirectory: {s2_out}")
    print(f"   S3OutputDirectory: {s3_out}")
    print(f"   S4OutputDirectory: {s4_out}\n")


def run_single_task(top_folder: Path):
    """ExecuteRowSinglefilesFolderFull pipeline"""
    try:
        print("=" * 70)
        print(f"🚀 StartTask: [{top_folder.name}]")
        print("=" * 70)

        # ConcatenateInsideLayer S1-pcc Directory + dataFile
        data_inner_dir = top_folder / DATA_SUB_FOLDER
        data_path = data_inner_dir / DATA_FILE

        # ValidateDirectoryandFile
        if not data_inner_dir.is_dir():
            print(f"⚠ WARNING: NotFoundInsideLayerDirectory {DATA_SUB_FOLDER},SkipThis task\n")
            return
        if not data_path.exists():
            print(f"⚠ WARNING: {data_inner_dir} InsideNotFound {DATA_FILE},SkipThis task\n")
            return

        # ModifyConfiguration
        modify_config(top_folder)

        # Call run.py RunRowFull pipeline
        cmd = ["python", str(RUN_SCRIPT)]
        result = subprocess.run(
            cmd,
            cwd=CODE_ROOT,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode == 0:
            print(f"🎉 Task[{top_folder.name}]Execution complete\n")
        else:
            print(f"❌ Task[{top_folder.name}]RunRowException,ReturnsCode: {result.returncode}\n")

    except Exception as e:
        print(f"❌ Task[{top_folder.name}]OutputWrong: {str(e)}\n")


def load_whitelist(csv_path: str) -> list:
    """FromCSVReadPendingRowFileFolderName(SingleColumn)"""
    import csv
    folder_list = []
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"WhitelistFileNotExists: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                folder_list.append(row[0].strip())
    return folder_list


def main():
    # 1. Backup originalConfiguration
    backup_original_config()

    # 2. ReadWhitelistFileFolderColumnTable
    try:
        white_folders = load_whitelist(WHITELIST_CSV)
    except Exception as e:
        print(f"ReadWhitelistFailed: {e}")
        return

    if not white_folders:
        print("⚠ WhitelistCSVInNoneHasEffectiveFileFolderName,Program exitOutput")
        return

    print(f"📋 FromWhitelistReadTo {len(white_folders)} PendingRowFileFolder:")
    for name in white_folders:
        print(f"   - {name}")
    print("-" * 70 + "\n")

    # 3. ConcatenateCompletePathAndValidate
    task_paths = []
    data_root = Path(DATA_ROOT)
    for folder_name in white_folders:
        full_path = data_root / folder_name
        if full_path.is_dir():
            task_paths.append(full_path)
        else:
            print(f"✘ DirectoryNotExists,Skip: {folder_name}")

    if not task_paths:
        print("\n⚠ AllWhitelistDirectoryAverageNoneEffective,Program exitOutput")
        restore_original_config()
        return

    # 4. StepExecuteRowTask
    total = len(task_paths)
    for idx, task_dir in enumerate(task_paths, 1):
        print(f"========== Progress {idx}/{total} ==========")
        run_single_task(task_dir)

    # 5. AllRun completeRestoreOriginalConfiguration
    restore_original_config()
    print("\n✅ AllBatchTaskExecuteRowCompleted!")


if __name__ == "__main__":
    main()