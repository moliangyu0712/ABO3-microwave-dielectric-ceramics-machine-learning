from __future__ import annotations
import contextlib
import io
import sys
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import ExpCellFilter
from ase.filters import FrechetCellFilter
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

# From matgl ImportOriginal TrajectoryObserver and Potential Class
from matgl.apps.pes import Potential
from matgl.ext.ase import OPTIMIZERS, PESCalculator, TrajectoryObserver
from tqdm import tqdm

if TYPE_CHECKING:
    from ase.optimize.optimize import Optimizer

# matgl = 1.2.1
# matminer = 0.7.4
# matplotlib = 3.8.4
# mp-api = 0.45.1
# numpy = 1.26.4
# pandas = 2.2.3
# torch = 2.2.0
# tqdm = 4.67.1
# pymatgen = 2024.8.9
# dgl = 1.1.2+cu118

# --- CustomException ---
class EarlyStoppingException(Exception):
    """Custom exception to break out of optimizer when early stopping condition is met."""
    pass


# --- CustomObserver ---
class EarlyStoppingObserver(TrajectoryObserver):
    """
    A custom observer inheriting from TrajectoryObserver.
    It adds early stopping when forces show no improvement over time.
    """
    def __init__(self, atoms: Atoms, patience: int, verbose: bool = False):
        """
        Initialize observer.

        Args:
            atoms (Atoms): Observed atomic structure.
            patience (int): Number of steps to wait when max force hasn't decreased.
            verbose (bool): Whether to print extra info in terminal.
        """
        super().__init__(atoms)  # Initialize parent class to save trajectory
        self.patience = patience
        self.verbose = verbose
        self.steps_since_best = 0
        self.best_force = float('inf')
        self.best_force_step_index = -1

    def __call__(self) -> None:
        """
        This method is called at each optimization step.
        It saves current state and checks if early stopping condition is met.
        """
        # 1. Call parent method to record energy, forces, positions, etc., ensuring complete trajectory
        super().__call__()

        # 2. CalculateMax force at current step (consistent with ASE optimizer fmax calculation)
        current_max_force = np.sqrt((self.atoms.get_forces()**2).sum(axis=1).max())

        # 3. Check if force has improved
        if current_max_force < self.best_force:
            self.best_force = current_max_force
            self.steps_since_best = 0
            # Record index of step with minimum force in trajectory list
            self.best_force_step_index = len(self.energies) - 1
            if self.verbose:
                # Print info, which will also be captured by your main program
                print(f"  Observer: New best force at step {self.best_force_step_index}: {self.best_force:.4f} eV/Å", flush=True)
        else:
            self.steps_since_best += 1

        # 4. IfExceed“Patience”StepsNoImprove,raise exception to stop optimization
        if self.steps_since_best >= self.patience:
            if self.verbose:
                print(f"\nObserver: Stopping early. Force has not improved for {self.patience} steps.", flush=True)
            raise EarlyStoppingException("Early stopping criterion met.")


# --- New Relaxer class with early stopping functionality ---
class Relaxer:
    """
    This is a local Relaxer class that overrides the version in matgl library.
    It addsOne'early_stop_patience'ParametersTo achieveFrontStop.
    """
    def __init__(
        self,
        potential: Potential,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 1 / 160.21766208,
    ):
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.calculator = PESCalculator(
            potential=potential,
            stress_weight=stress_weight,
        )
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms | Structure | Molecule,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        verbose: bool = False,
        ase_cellfilter: Literal["Frechet", "Exp"] = "Frechet",
        params_asecellfilter: dict | None = None,
        early_stop_patience: int | None = None,  # <-- KeyNewly addedParameters
        **kwargs,
    ):
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            # Choose different observer based on whether early stopping is needed
            if early_stop_patience is not None:
                obs = EarlyStoppingObserver(atoms, patience=early_stop_patience, verbose=verbose)
            else:
                obs = TrajectoryObserver(atoms)

            if self.relax_cell:
                cell_filter_params = params_asecellfilter or {}
                atoms = (
                    FrechetCellFilter(atoms, **cell_filter_params)
                    if ase_cellfilter == "Frechet"
                    else ExpCellFilter(atoms, **cell_filter_params)
                )

            optimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)

            try:
                # RunRowOptimize,IfTriggerFrontStop,EarlyStoppingExceptionWillInThrown hereOutput
                optimizer.run(fmax=fmax, steps=steps)
            except EarlyStoppingException as e:
                if verbose:
                    print(f"Relaxation interrupted by early stopping logic: {e}", flush=True)

            # EnsureMostBackOneStepStatusAlsoRecordBottomTo
            obs()

        if traj_file is not None:
            obs.save(traj_file)
            
        final_atoms_obj = atoms.atoms if isinstance(atoms, (FrechetCellFilter, ExpCellFilter)) else atoms

        # CheckRelaxationIsHow it ended
        if isinstance(obs, EarlyStoppingObserver) and obs.steps_since_best >= obs.patience:
            print("\nRelaxation finished due to early stopping. Returning the structure with the minimum force found.", flush=True)
            # From trajectoryInRebuild forceMinimumThatOneStepAtomstructures
            best_step_idx = obs.best_force_step_index
            best_atoms = Atoms(
                numbers=final_atoms_obj.get_atomic_numbers(),
                positions=obs.atom_positions[best_step_idx],
                cell=obs.cells[best_step_idx],
                pbc=final_atoms_obj.pbc,
            )
            final_structure = self.ase_adaptor.get_structure(best_atoms)
        else:
            print("\nRelaxation finished normally (reached fmax or max steps). Returning the final structure.", flush=True)
            final_structure = self.ase_adaptor.get_structure(final_atoms_obj)

        return {
            "final_structure": final_structure,
            "trajectory": obs,
        }


# ImportLibrary
import re
import warnings
import os
import time
# from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import matgl
from pymatgen.io.cif import CifWriter
warnings.simplefilter("ignore")
from glob import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
# from pymatgen.core import Structure
warnings.simplefilter("ignore")

def init_worker():
    """InitializeWorker process,OnlyLoadOneTimeModel"""
    global pot, relaxer
    print(f"Process {os.getpid()} PositiveInLoadModel...")
    try:
        pot = matgl.load_model("D:/MLY/M3GNet-MP-2021.2.8-PES/")
        relaxer = Relaxer(potential=pot)
        print(f"Process {os.getpid()} ModelLoadCompleted")
    except Exception as e:
        print(f"Process {os.getpid()} Model loading failed: {e}")
        # IfModel loading failed,SetrelaxerAsNone
        relaxer = None


# Wrap main loop code into a function
def process_single_file(data):
    try:
        time1 = time.time()
        name = os.path.basename(data)[:-4]
        print(f"Computing: {name}")
        
        # Set output path
        output_path = "D:/MLY/ABO3/data/validation_structures/11/"
        struc_file = f"{output_path}{name}/"
        if not os.path.exists(struc_file):
            os.makedirs(struc_file)
        # GetGenerateCIFFile
        parser = CifParser(data)
        structures_0 = parser.parse_structures(primitive=False, check_occu=False)
        structure = structures_0[0]

        log_file_path = os.path.join(struc_file, 'relaxation.log')    
        # SetTrajectoryFilePath
        traj_file_path = os.path.join(struc_file, 'relaxation_trajectory.pkl')
        
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            with contextlib.redirect_stdout(log_file):
                # ProceedRelaxation
                relax_results = relaxer.relax(
                    structure,
                    fmax=0.05,
                    steps = 10000,
                    early_stop_patience=5000,
                    traj_file=traj_file_path,
                    interval=1,
                    verbose=True
                )
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            output_text = f.read()
        
        # ParseOutput
        steps = []
        times = []
        energies = []
        fmax_values = []
        
        pattern = r'FIRE:\s*(\d+)\s+(\d+:\d+:\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
        matches = re.findall(pattern, output_text)
        
        for match in matches:
            step, time_str, energy_str, fmax_str = match
            steps.append(int(step))
            times.append(time_str)
            energies.append(float(energy_str))
            fmax_values.append(float(fmax_str))
        
        # CreateDataFrame
        relaxation_df = pd.DataFrame({
            'Step': steps,
            'Time': times,
            'Energy': energies,
            'fmax': fmax_values
        })
        
        # SaveToCSVFile
        csv_path = os.path.join(struc_file, 'relaxation_progress.csv')
        relaxation_df.to_csv(csv_path, index=False)
        
        time2 = time.time()
        time3 = time2 - time1
        
        # SaveTotalResultInformation
        summary_lines = [
            f"Relaxation process data saved to: {csv_path}",
            f"Total relaxation steps: {len(relaxation_df)}",
            f"Initial energy: {relaxation_df['Energy'].iloc[0]:.6f} eV",
            f"Final energy: {relaxation_df['Energy'].iloc[-1]:.6f} eV",
            f"Energy change: {relaxation_df['Energy'].iloc[-1] - relaxation_df['Energy'].iloc[0]:.6f} eV",
            f"Final fmax: {relaxation_df['fmax'].iloc[-1]:.6f}",
            f"Time elapsed: {time3:.2f} s"
        ]
        
        summary_text = "\n".join(summary_lines)
        
        # WriteTXTFile
        txt_save_path = os.path.join(struc_file, "relaxation_summary.txt")
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        # SaveFinalstructures
        final_structure = relax_results["final_structure"]
        final_structure_path = os.path.join(struc_file, 'final_structure.cif')
        CifWriter(final_structure).write_file(final_structure_path)
        
        # DrawRelaxationProcessFigureTable
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(relaxation_df['Step'], relaxation_df['Energy'], 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'Energy vs Relaxation Step - {name}')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(relaxation_df['Step'], relaxation_df['fmax'], 'r-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('fmax')
        ax2.set_title(f'Maximum Force vs Relaxation Step - {name}')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(struc_file, 'relaxation_plot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)  # ClosePlot,AvoidMemoryLeak
    
        print(f"Completed: {name}, Time elapsed: {time3:.2f} s")
        return name, time3
    except Exception as e:
        print(f"!!! ErrorCalculate{name}WhenSeriousError{e}")
        return name, "FAILED"
            

def main():
    # Get all CIF files
    datas = glob("D:/MLY/ABO3/data/validation_structures/11/*.cif")

    print(f"Found {len(datas)} files")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Set process count (recommended: CPU cores - 1, leave one for system)
    total_cores = os.cpu_count()
    # num_processes = max(1, int(total_cores * 0.75))
    num_processes = 5
    print(f"System cores:{total_cores},Using {num_processes} processes for parallel computation")
    
    start_time = time.time()
    
    # Using Pool for parallel computation
    with Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=5) as pool:
        
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
