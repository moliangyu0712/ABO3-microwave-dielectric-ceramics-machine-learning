from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells, _get_sqs_cluster_vector

import csv
import logging
import numpy as np
import os
import pandas as pd
import re

from typing import Tuple

from GNBO.crystal.superlattice import make_supercell

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

current_dir = os.path.dirname(os.path.abspath(__file__))

class IcetScoreCaptureHandler(logging.Handler):
    """From icet LogInCapture current score and running best score."""

    def __init__(self):
        super().__init__()
        self.best_score = None
        self.current_score = None
        self.history = []

        number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

        self.step_pattern = re.compile(r"\bstep\s*[:=]?\s*(\d+)", re.IGNORECASE)
        self.best_score_pattern = re.compile(rf"\bbest\s+score\s*[:=]\s*{number}", re.IGNORECASE)

        self.current_score_patterns = [
            re.compile(rf"\bcurrent\s+score\s*[:=]\s*{number}", re.IGNORECASE),
            re.compile(rf"\btrial\s+score\s*[:=]\s*{number}", re.IGNORECASE),
            re.compile(rf"\baccepted\s+score\s*[:=]\s*{number}", re.IGNORECASE),
        ]

    def reset(self):
        self.best_score = None
        self.current_score = None
        self.history = []

    def emit(self, record):
        log_message = self.format(record)

        step = None
        step_match = self.step_pattern.search(log_message)
        if step_match:
            step = int(step_match.group(1))

        current_score = None
        for pattern in self.current_score_patterns:
            current_match = pattern.search(log_message)
            if current_match:
                current_score = float(current_match.group(1))
                self.current_score = current_score
                break

        best_score = None
        best_match = self.best_score_pattern.search(log_message)
        if best_match:
            best_score = float(best_match.group(1))
            self.best_score = best_score

        if current_score is not None or best_score is not None:
            self.history.append({
                "step": step,
                "current_score": current_score,
                "best_score": best_score,
            })

    def get_step_score_map(self):
        """
        Returns:
            {
                step: {
                    "current_score": float or None,
                    "best_score": float or None,
                }
            }

        IfLogNo step Number,ThenByLogAppearance order as step.
        """
        step_score_map = {}

        for index, item in enumerate(self.history):
            step = item["step"] if item["step"] is not None else index

            if step not in step_score_map:
                step_score_map[step] = {
                    "current_score": None,
                    "best_score": None,
                }

            if item["current_score"] is not None:
                step_score_map[step]["current_score"] = item["current_score"]

            if item["best_score"] is not None:
                step_score_map[step]["best_score"] = item["best_score"]

        return step_score_map

class SQSOptimizer:
    """
    SQSMethodReference paper: 
    ICET – A Python Library for Constructing and Sampling Alloy Cluster Expansions
    DOI: https://doi.org/10.1002/adts.201900015
    CorrespondingUse website: https://icet.materialsmodeling.org/advanced_topics/sqs_generation.html

    Parameters: 
        strict_concentration:
            True  - StrictUsingInputstructuresInpreset ofconcentration
            False - Auto fine-tuningconcentration,Useconcentration×supercellSiteCountAsinteger
    """

    def __init__(
        self,
        structure: Structure = None,
        max_cluster: int = 4,
        output: str = current_dir,
        MC_step: int = 3000,
        verbose: bool = False,
        strict_concentration: bool = True,
    ):
        self.structure = structure
        self.output = output
        self.max_cluster = max_cluster
        self.verbose = verbose
        self.MC_step = MC_step
        self.strict_concentration = strict_concentration

        os.makedirs(self.output, exist_ok=True)

        self.log_file = os.path.join(self.output, "sqs_log.txt")
        self.step_scores_csv = os.path.join(self.output, "icet_step_scores.csv")
        self.result_csv = os.path.join(self.output, "sqs_result.csv")

        self.primitive_structure = self.get_primitive_structure()

        score_capture_handler = IcetScoreCaptureHandler()

        with open(self.log_file, "w", encoding="utf-8"):
            pass

        with open(self.step_scores_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "supercell",
                "step",
                "current_icet_score",
                "running_best_icet_score",
            ])

        adaptor = AseAtomsAdaptor()

        ase_primitive = adaptor.get_atoms(self.primitive_structure)
        ase_primitive.pbc = [True, True, True]

        scaled_pos = ase_primitive.get_scaled_positions()
        scaled_pos = scaled_pos % 1
        scaled_pos = np.round(scaled_pos, 4)
        ase_primitive.set_scaled_positions(scaled_pos)

        chemical_symbols, target_concentrations, sublattice_site_counts = self._get_chemical_symbols()

        sqs_result = {}
        cutoff = self._find_cutoff(tuple([3, 3, 3]), 0.5)

        for i in range(5, 6):
            self._log_message(f"=====================  StartProceedsupercell{i}{i}{i}Calculate  =====================")

            supercell_matrix = [i, i, i]
            ase_supercell = ase_primitive.repeat(supercell_matrix)

            score_capture_handler.reset()
            self.configure_icet_logging(self.log_file, score_capture_handler)

            try:
                if self.strict_concentration:
                    calibrated_concentrations = target_concentrations
                    self._log_message(f"[StrictconcentrationMode]Usingoriginal concentration: {calibrated_concentrations}")
                else:
                    calibrated_concentrations = self._calibrate_concentrations(
                        supercell_size=i,
                        original_concentrations=target_concentrations,
                        sublattice_site_counts=sublattice_site_counts,
                    )
                    self._log_message(f"[non-strict concentration mode]using calibrated concentration: {calibrated_concentrations}")

                cs = ClusterSpace(
                    ase_primitive,
                    cutoffs=cutoff,
                    chemical_symbols=chemical_symbols,
                )

                target_cv = _get_sqs_cluster_vector(cs, calibrated_concentrations)

                sqs_from_super = generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=[ase_supercell],
                    target_concentrations=calibrated_concentrations,
                    random_seed=42,
                    n_steps=self.MC_step,
                    tol=1e-4,
                )

                gen_cv = cs.get_cluster_vector(sqs_from_super)
                mae, rmse = self._calculate_cv_metrics(target_cv, gen_cv)

                self._log_message(f"Target cluster vector: {target_cv.tolist()}")
                self._log_message(f"Cluster vector of generated structure: {gen_cv.tolist()}")
                self._log_message(f"Final MAE: {mae:.10f}")
                self._log_message(f"Final RMSE: {rmse:.10f}")

                pmg_sqs = adaptor.get_structure(sqs_from_super)

                final_cv = cs.get_cluster_vector(sqs_from_super)
                for idx, orbit in enumerate(cs.as_list):
                    orbit_order = orbit["order"]
                    cv_value = final_cv[idx]
                    self._log_message(f"{orbit_order}-tuple cluster contribution: {cv_value}")

                icet_score = score_capture_handler.best_score
                if icet_score is None:
                    icet_score = float("inf")
                    self._log_message("WARNING: NotFrom icet LogCaptured best score,Currentsupercell icet_score NoteAs inf")

                self._write_step_scores_csv(
                    supercell_size=i,
                    score_handler=score_capture_handler,
                )

                sqs_result[str(int(i))] = {
                    "icet_score": icet_score,
                    "mae": mae,
                    "rmse": rmse,
                    "structure": pmg_sqs,
                }

                if self.verbose:
                    output_structure_dir = os.path.join(self.output, "sqs_structure")
                    os.makedirs(output_structure_dir, exist_ok=True)

                    CifWriter(pmg_sqs).write_file(
                        os.path.join(
                            output_structure_dir,
                            f"{i}_{i}_{i}_icet_{icet_score:.6f}_mae_{mae:.6f}_rmse_{rmse:.6f}.cif",
                        )
                    )

                self._write_result_summary(sqs_result)

            except Exception as e:
                self._log_message(f"{[i, i, i]}supercellCalculateFailed: {str(e)}")

        if sqs_result:
            global_best_number = min(sqs_result, key=lambda k: sqs_result[k]["icet_score"])

            global_best = sqs_result[global_best_number]
            global_best_score = global_best["icet_score"]
            global_best_mae = global_best["mae"]
            global_best_rmse = global_best["rmse"]
            global_best_structure = global_best["structure"]

            output_filename = (
                f"{global_best_number}{global_best_number}{global_best_number}"
                f"_icet_{global_best_score:.6f}"
                f"_mae_{global_best_mae:.6f}"
                f"_rmse_{global_best_rmse:.6f}.cif"
            )

            CifWriter(global_best_structure).write_file(
                os.path.join(self.output, output_filename)
            )

            self._log_message(
                f"OptimalstructuresAlreadySave: {output_filename} | "
                f"icet_score={global_best_score:.10f}, "
                f"mae={global_best_mae:.10f}, "
                f"rmse={global_best_rmse:.10f}"
            )
        else:
            self._log_message("AllsupercellCalculateAverageFailed,NotGenerateOptimalstructures")

    def _calculate_cv_metrics(self, target_cv, gen_cv):
        target_cv = np.asarray(target_cv, dtype=np.float64)
        gen_cv = np.asarray(gen_cv, dtype=np.float64)

        diff = target_cv - gen_cv
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        return mae, rmse

    def _write_step_scores_csv(self, supercell_size, score_handler):
        step_score_map = score_handler.get_step_score_map()

        last_current_score = None
        last_best_score = None

        with open(self.step_scores_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            for step in range(self.MC_step):
                if step in step_score_map:
                    current_score = step_score_map[step]["current_score"]
                    best_score = step_score_map[step]["best_score"]

                    if current_score is not None:
                        last_current_score = current_score

                    if best_score is not None:
                        last_best_score = best_score

                writer.writerow([
                    f"{supercell_size}{supercell_size}{supercell_size}",
                    step,
                    "" if last_current_score is None else last_current_score,
                    "" if last_best_score is None else last_best_score,
                ])

    def _write_result_summary(self, sqs_result):
        rows = []

        for key, value in sqs_result.items():
            rows.append({
                "super_lattice": [int(key), int(key), int(key)],
                "icet_score": value["icet_score"],
                "mae": value["mae"],
                "rmse": value["rmse"],
            })

        result = pd.DataFrame(rows)
        result.to_csv(self.result_csv, index=False)

    def _get_chemical_symbols(self):
        """
        return:
            chemical_symbols:
                [['Ba', 'Sr', 'Ca'], ['Hf'], ['O'], ['O'], ['O']]
            target_concentrations:
                {'A': {'Ba': 0.75, 'Sr': 0.125, 'Ca': 0.125}, ...}
                Single activesublatticeWhenReturns {'Ba': 0.75, ...}
            sublattice_site_counts:
                {'A': 1, 'B': 2, ...}
                Single activesublatticeWhenReturns int
        """
        site_info = []
        active_symbol_groups = []

        for site in self.structure:
            species_str = site.species_string

            if len(species_str) > 4 and ":" in species_str:
                atoms = [item.split(":")[0] for item in species_str.split(", ")]
                ratios = [float(item.split(":")[1]) for item in species_str.split(", ")]
                concentration = dict(zip(atoms, ratios))
                sorted_atoms = sorted(atoms)

                site_info.append((True, sorted_atoms, concentration))
                active_symbol_groups.append(tuple(sorted_atoms))
            else:
                site_info.append((False, [species_str], None))

        unique_active_groups = list(set(active_symbol_groups))
        unique_active_groups.sort(key=lambda group: [elem for elem in group])

        group_to_sublattice = {
            group: chr(65 + idx)
            for idx, group in enumerate(unique_active_groups)
        }

        sublattice_site_counts = {}
        for group in unique_active_groups:
            sublattice = group_to_sublattice[group]
            count = sum(
                1
                for is_active, symbols, _ in site_info
                if is_active and tuple(sorted(symbols)) == group
            )
            sublattice_site_counts[sublattice] = count

        chemical_symbols = []
        for is_active, symbols, _ in site_info:
            if is_active:
                sorted_group = tuple(sorted(symbols))
                chemical_symbols.append(list(sorted_group))
            else:
                chemical_symbols.append(symbols)

        target_concentrations = {}
        group_concentrations = {}

        for is_active, symbols, conc in site_info:
            if is_active and conc is not None:
                sorted_group = tuple(sorted(symbols))
                if sorted_group not in group_concentrations:
                    group_concentrations[sorted_group] = conc

        for sorted_group in unique_active_groups:
            sublattice_symbol = group_to_sublattice[sorted_group]
            target_concentrations[sublattice_symbol] = group_concentrations[sorted_group]

        if len(target_concentrations) == 1:
            target_concentrations = list(target_concentrations.values())[0]
            sublattice_site_counts = list(sublattice_site_counts.values())[0]

        return chemical_symbols, target_concentrations, sublattice_site_counts

    def _calibrate_concentrations(self, supercell_size: int, original_concentrations, sublattice_site_counts):
        def _handle_single_element(elements, total_sites):
            if len(elements) == 1:
                atom_count = total_sites
                conc = round(atom_count / total_sites, 8)
                self._log_message(
                    f"SingleElementcalibration: AtomCount={atom_count}(total sites={total_sites}),concentration={conc}"
                )
                return {elements[0]: conc}, [atom_count]

            return None

        if isinstance(original_concentrations, dict) and all(
            isinstance(v, float) for v in original_concentrations.values()
        ):
            elements = list(original_concentrations.keys())
            n_elem = len(elements)
            total_sites = sublattice_site_counts * (supercell_size ** 3)

            self._log_message(f"\n=== supercell{supercell_size}×{supercell_size}×{supercell_size} calibration ===")
            self._log_message(f"original concentration: {original_concentrations} | supercelltotal sites: {total_sites}")

            single_result = _handle_single_element(elements, total_sites)
            if single_result:
                return single_result[0]

            original_atoms = np.array(
                [original_concentrations[elem] * total_sites for elem in elements],
                dtype=np.float64,
            )
            self._log_message(f"Original atom count(Decimal): {dict(zip(elements, original_atoms))}")

            target_atoms = np.zeros(n_elem, dtype=np.int64)

            for idx in range(n_elem - 1):
                target_atoms[idx] = np.round(original_atoms[idx]).astype(np.int64)

            target_atoms[-1] = total_sites - np.sum(target_atoms[:-1])
            target_atoms = self._repair_atom_counts(elements, target_atoms, total_sites)

            calibrated = {}
            for elem, atoms in zip(elements, target_atoms):
                conc = atoms / total_sites
                calibrated[elem] = round(conc, 8)

            self._log_message(
                f"calibrated atom counts(integer): {dict(zip(elements, target_atoms))}(sum={np.sum(target_atoms)})"
            )
            self._log_message(
                f"calibrated concentration: {calibrated}(sum={sum(calibrated.values()):.8f})"
            )

            for elem, conc in calibrated.items():
                atom_verify = int(round(conc * total_sites))
                self._log_message(
                    f"verification: {elem}concentration={conc} × total sites={total_sites} = {atom_verify}(integer)"
                )

            return calibrated

        if isinstance(original_concentrations, dict) and all(
            isinstance(v, dict) for v in original_concentrations.values()
        ):
            calibrated = {}

            for sublattice, conc_dict in original_concentrations.items():
                elements = list(conc_dict.keys())
                n_elem = len(elements)
                total_sites = sublattice_site_counts[sublattice] * (supercell_size ** 3)

                self._log_message(
                    f"\n=== sublattice{sublattice} supercell{supercell_size}×{supercell_size}×{supercell_size} calibration ==="
                )
                self._log_message(f"original concentration: {conc_dict} | total sublattice sites: {total_sites}")

                single_result = _handle_single_element(elements, total_sites)
                if single_result:
                    calibrated[sublattice] = single_result[0]
                    continue

                original_atoms = np.array(
                    [conc_dict[elem] * total_sites for elem in elements],
                    dtype=np.float64,
                )

                target_atoms = np.zeros(n_elem, dtype=np.int64)

                for idx in range(n_elem - 1):
                    target_atoms[idx] = np.round(original_atoms[idx]).astype(np.int64)

                target_atoms[-1] = total_sites - np.sum(target_atoms[:-1])
                target_atoms = self._repair_atom_counts(elements, target_atoms, total_sites)

                sub_calibrated = {}
                for elem, atoms in zip(elements, target_atoms):
                    conc = atoms / total_sites
                    sub_calibrated[elem] = round(conc, 8)

                self._log_message(
                    f"sublattice{sublattice}calibrated atom counts: {dict(zip(elements, target_atoms))}(sum={np.sum(target_atoms)})"
                )
                self._log_message(
                    f"sublattice{sublattice}calibrated concentration: {sub_calibrated}(sum={sum(sub_calibrated.values()):.8f})"
                )

                for elem, conc in sub_calibrated.items():
                    atom_verify = int(round(conc * total_sites))
                    self._log_message(
                        f"sublattice{sublattice}verification: {elem}concentration={conc} × total sites={total_sites} = {atom_verify}(integer)"
                    )

                calibrated[sublattice] = sub_calibrated

            return calibrated

        self._log_message(f"NotSupportconcentrationFormat,Skipcalibration: {type(original_concentrations)}")
        return original_concentrations

    def _repair_atom_counts(self, elements, target_atoms, total_sites):
        neg_indices = np.where(target_atoms < 0)[0]

        if len(neg_indices) > 0:
            self._log_message(f"Find negativeAtomCount: {dict(zip(elements, target_atoms))},StartCorrect")

            for neg_idx in neg_indices:
                need_add = abs(target_atoms[neg_idx])
                max_atom_idx = np.argmax(target_atoms)

                if target_atoms[max_atom_idx] > need_add:
                    target_atoms[max_atom_idx] -= need_add
                    target_atoms[neg_idx] = 0
                else:
                    non_neg_indices = np.where(target_atoms > 0)[0]

                    if len(non_neg_indices) == 0:
                        target_atoms[:] = 0
                        target_atoms[-1] = total_sites
                        return target_atoms

                    each_sub = need_add // len(non_neg_indices)
                    remain_sub = need_add % len(non_neg_indices)

                    for local_idx, idx in enumerate(non_neg_indices):
                        sub_val = each_sub + (1 if local_idx < remain_sub else 0)
                        target_atoms[idx] = max(0, target_atoms[idx] - sub_val)

                    target_atoms[neg_idx] = 0

        target_atoms = np.maximum(target_atoms, 0)
        target_atoms[-1] = total_sites - np.sum(target_atoms[:-1])

        if target_atoms[-1] < 0:
            excess = abs(target_atoms[-1])
            target_atoms[-1] = 0

            for idx in np.argsort(target_atoms)[::-1]:
                if excess <= 0:
                    break

                reduction = min(target_atoms[idx], excess)
                target_atoms[idx] -= reduction
                excess -= reduction

            target_atoms[-1] = total_sites - np.sum(target_atoms[:-1])

        return target_atoms

    def _find_cutoff(self, scaling_matrix: Tuple[int, int, int], cutoff_gap: float = 0.5) -> Tuple[float, float, float]:
        supercell = make_supercell(self.structure, scaling_matrix)

        self._log_message("\n== Auto cutoff search started ==")
        self._log_message(f"supercell expansion matrix: {scaling_matrix},total atoms in supercell: {len(supercell)}")

        dopant_coords = []

        for site in supercell:
            if len(site.species_string) > 4:
                dopant_coords.append(site.coords)

        if len(dopant_coords) < 2:
            raise ValueError(f"Doping siteAtomCountNotSufficient(Only{len(dopant_coords)}),NoneMethodCalculateSpacing")

        dopant_coords = np.array(dopant_coords, dtype=np.float64)

        distances = []
        n_atoms = len(dopant_coords)

        for idx_i in range(n_atoms):
            for idx_j in range(idx_i + 1, n_atoms):
                dist = np.linalg.norm(dopant_coords[idx_j] - dopant_coords[idx_i])
                if dist > 1e-5:
                    distances.append(round(dist, 4))

        distances = sorted(list(set(distances)))

        if len(distances) == 0:
            raise ValueError("NotCalculateTo existingEffectiveDoping siteAtom spacing")

        if len(distances) < 2:
            raise ValueError("HasEffectiveDoping siteSpacing less than2,NoneMethodRecognize2nd nearest neighbor")

        self._log_message(f"doping site atomic distance range: {distances[0]:.4f} ~ {distances[-1]:.4f} Å")
        self._log_message(f"all unique distances(top 10): {distances[:10]}")

        first_neighbor = distances[0]
        second_neighbor = distances[1]

        self._log_message("identified key neighbor distances: ")
        self._log_message(f"  - 1st nearest neighbor: {first_neighbor:.4f} Å")
        self._log_message(f"  - 2nd nearest neighbor: {second_neighbor:.4f} Å")

        cutoff = []
        cutoff_radius = 0

        for idx in range(1, self.max_cluster):
            cluster_order = idx + 1

            if cluster_order == 2:
                cutoff_radius = round(second_neighbor + cutoff_gap, 2)
                self._log_message(f"  - {cluster_order}-tuple cluster: {cutoff_radius:.2f} Å")
                cutoff.append(cutoff_radius)

            if cluster_order > 2:
                cutoff_radius = cutoff_radius - cutoff_gap

                if cutoff_radius < first_neighbor:
                    self._log_message("== Auto cutoff search ended ==\n")
                    self._log_message(
                        f"WARNING: {cluster_order}-tuple cluster search range is already smaller than1st nearest neighbor,recommended max cluster dimension is{idx}"
                    )
                    return cutoff

                self._log_message(f"  - {cluster_order}-tuple cluster: {cutoff_radius:.2f} Å")
                cutoff.append(cutoff_radius)

        self._log_message("== Auto cutoff search ended ==\n")
        return cutoff

    def get_primitive_structure(self):
        analyzer = SpacegroupAnalyzer(self.structure)
        sg = analyzer.get_space_group_number()
        symmetrized_structure = analyzer.get_symmetrized_structure()

        lattice_list = {
            "a": symmetrized_structure.lattice.a,
            "b": symmetrized_structure.lattice.b,
            "c": symmetrized_structure.lattice.c,
            "alpha": symmetrized_structure.lattice.alpha,
            "beta": symmetrized_structure.lattice.beta,
            "gamma": symmetrized_structure.lattice.gamma,
        }

        lattice_params = self._get_lattice_parameters(sg, lattice_list)
        lattice = Lattice.from_parameters(**lattice_params)

        primitive_struct = Structure(
            lattice=lattice,
            species=[],
            coords=[],
            to_unit_cell=True,
        )

        for ele in symmetrized_structure:
            frac_coord = ele.frac_coords

            if len(ele.species_string) > 4:
                atom = ele.species_string.split(", ")[0].split(":")[0]
            else:
                atom = ele.species_string

            primitive_struct.append(
                species=atom,
                coords=frac_coord,
                properties={"occupancy": 1},
            )

        return primitive_struct

    def _get_lattice_parameters(self, sg, struc_parameters):
        if sg in [0, 1, 2]:
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["b"],
                "c": struc_parameters["c"],
                "alpha": struc_parameters["alpha"],
                "beta": struc_parameters["beta"],
                "gamma": struc_parameters["gamma"],
            }
        elif sg in list(range(3, 15 + 1)):
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["b"],
                "c": struc_parameters["c"],
                "alpha": 90,
                "beta": struc_parameters["beta"],
                "gamma": 90,
            }
        elif sg in list(range(16, 74 + 1)):
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["b"],
                "c": struc_parameters["c"],
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
            }
        elif sg in list(range(75, 142 + 1)):
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["a"],
                "c": struc_parameters["c"],
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
            }
        elif sg in list(range(143, 194 + 1)):
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["a"],
                "c": struc_parameters["c"],
                "alpha": 90,
                "beta": 90,
                "gamma": 120,
            }
        elif sg in list(range(195, 230 + 1)):
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["a"],
                "c": struc_parameters["a"],
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
            }
        else:
            base_params = {
                "a": struc_parameters["a"],
                "b": struc_parameters["b"],
                "c": struc_parameters["c"],
                "alpha": struc_parameters["alpha"],
                "beta": struc_parameters["beta"],
                "gamma": struc_parameters["gamma"],
            }

        return base_params

    def _log_message(self, message: str):
        os.makedirs(self.output, exist_ok=True)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def configure_icet_logging(self, log_file_path, score_handler):
        icet_logger = logging.getLogger("icet")
        icet_logger.setLevel(logging.INFO)

        for handler in icet_logger.handlers[:]:
            icet_logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(name)s: %(levelname)s  %(message)s"))

        score_handler.setFormatter(logging.Formatter("%(message)s"))

        icet_logger.addHandler(file_handler)
        icet_logger.addHandler(score_handler)