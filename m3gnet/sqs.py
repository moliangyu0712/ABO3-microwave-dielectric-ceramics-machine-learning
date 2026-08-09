from __future__ import annotations
from glob import glob
from tqdm import tqdm
from GNBO.sqs.sqs import SQSOptimizer
import os
import time
# from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from multiprocessing import Pool, cpu_count
import warnings

# from pymatgen.core import Structure
warnings.simplefilter("ignore")


# Wrap main loop code into a function
def process_single_file(data):
    try:
        time1 = time.time()
        name = os.path.basename(data)[:-4]
        print(f"Computing: {name}")
        
        # Set output path
        output_path = "D:/MLY/ABO3/data/structure_data_latest/example_structures/doping_examples/1/supplement/relaxed/"
        struc_file = f"{output_path}{name}/"
        
        # Read structure
        parser = CifParser(data)
        structures = parser.parse_structures(primitive=False, check_occu=False)
        structure = structures[0]
        
        # Create output directory
        if not os.path.exists(struc_file):
            os.makedirs(struc_file, exist_ok=True)
            
        if not os.path.exists(os.path.join(struc_file, "sqs_result.csv")):  
            # SQSOptimize
            SQSO = SQSOptimizer(
                structure=structure,
                output=struc_file,
                verbose=True,
                strict_concentration=False
            )
            time2 = time.time()
            time3 = time2 - time1
        
            print(f"Completed: {name}, Time elapsed: {time3:.2f} s")
            return name, time3
            
    except Exception as e:
        print(f"Processing {os.path.basename(data)} Error in: {str(e)}")
        return None
            

def main():
    # Get all CIF files
    datas = glob("D:/MLY/ABO3/data/structure_data_latest/example_structures/doping_examples/1/supplement/unrelaxed/*.cif")
    print(f"Found {len(datas)} files")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Set process count (recommended: CPU cores - 1, leave one for system)
    total_cores = os.cpu_count()
    # num_processes = max(1, int(total_cores * 0.75))
    num_processes = 5
    
    print(f"System cores:{total_cores},Using {num_processes} processes for parallel computation")
    
    start_time = time.time()
    
    # Using Pool for parallel computation
    with Pool(processes=num_processes, maxtasksperchild=5) as pool:
        
        # --- OptimizePoint 2: Using imap_unordered ImplementReal-timeProgressFeedback ---
        results = []
        total_tasks = len(datas)
        
        # pool.imap_unordered WillReturnsOneIteration,PerWhenOne taskCompletedGenerateOneResult
        # UsingtqdmLibrary can create a very nice progress bar
        try:
            task_iterator = tqdm(pool.imap_unordered(process_single_file, datas), total=total_tasks, desc="Relaxation progress")
        except ImportError:
            task_iterator = pool.imap_unordered(process_single_file, datas)

        # Real-timeProcessingAlreadyCompletedTask
        for i, result in enumerate(task_iterator):
            results.append(result)
            # IfNoInstalltqdm,CanTo manually printProgress
            # print(f"Progress: {i + 1}/{total_tasks} AlreadyCompleted.")

    end_time = time.time()
    
    # --- TaskCompletedBackSummary of ---
    print("\n" + "="*50)
    print("AllCalculateTaskCompleted!")
    
    successful_tasks = []
    failed_tasks = []
    
    for name, outcome in results:
        if outcome == "FAILED":
            failed_tasks.append(name)
        else:
            successful_tasks.append((name, outcome))
            
    print(f"\nTotalTime elapsed: {(end_time - start_time):.2f} s")
    print(f"SuccessTask count: {len(successful_tasks)}")
    print(f"FailedTask count: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\nFailedTaskColumnTable:")
        for name in failed_tasks:
            print(f"- {name}")
            
    print("="*50)


if __name__ == "__main__":
    main()
