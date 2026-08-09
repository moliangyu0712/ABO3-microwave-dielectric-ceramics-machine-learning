from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import numpy as np
import os
from pymatgen.core import Structure, Lattice, Composition
import re
from typing import Tuple

# FixWindowsBottomnumpy dtypeConflict
np.set_printoptions(precision=8)
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
class find_average_structure:
    def __init__(
            self,
            target: list = None,
            superlattice_relax: Structure = None,
            target_sp: int = None,
            output: str = current_dir
    ):
        
        self.superlattice_relax = superlattice_relax
        self.target_composition = target 
        self.output = output
        self.target_sp = target_sp
        # EnsureOutputDirectoryExists
        os.makedirs(self.output, exist_ok=True)

    def get_average_composition(self):
        new_structure = self.superlattice_relax.copy()
        for i, site in enumerate(self.superlattice_relax):
            for j, composition in enumerate(self.target_composition):
                composition_list = list(composition.keys())
                if site.species_string in composition_list:
                    new_structure[i].species = Composition(composition)
        return new_structure
    
    def get_symmetrized_structure(self, structure, symprec):
        analyzer = SpacegroupAnalyzer(structure=structure, symprec=symprec, angle_tolerance=10)
        sg = analyzer.get_space_group_number()
        symmetrized_structure = analyzer.get_symmetrized_structure()
        return symmetrized_structure, sg
        
    def get_refined_structure_from_cif(self, structure, symprec):
        analyzer = SpacegroupAnalyzer(structure=structure, symprec=symprec)
        symmetrized_structure = analyzer.get_refined_structure()
        return symmetrized_structure
    
    def get_refined_structure(self, structure_unrelax: Structure, sm: Tuple[int, int, int], write_cif: bool = False):
        ss = 0
        for i in self.superlattice_relax:
            ss = ss + 1
            if len(self.superlattice_relax[ss-1].species_string) > 4:
                break
        average_superlattice = self.get_average_structure(structure_unrelax, sm)

        k = 0
        symprec_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
        best_symmetry = 0
        best_symprec = 0
        for i in symprec_list:
            k += 1
            symmetrized_average_structure, sgn = self.get_symmetrized_structure(average_superlattice, i)      
            if sgn > 1:
                best_symmetry = sgn
                best_symprec = i
                if best_symmetry == self.target_sp:
                    print(f"Found target symmetry! Space group={best_symmetry}, symprec={best_symprec}")
                    refined_structure = self.get_refined_structure_from_cif(symmetrized_average_structure, best_symprec)
                    if write_cif:
                        CifWriter(refined_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                    return refined_structure
        if k == len(symprec_list):
            if best_symmetry != self.target_sp and best_symmetry != 0:
                print(f"Target symmetry not found!Best space group={best_symmetry}, Bestsymprec={best_symprec}")
                refined_structure = self.get_refined_structure_from_cif(symmetrized_average_structure, best_symprec)
                if write_cif:
                    CifWriter(refined_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                return refined_structure
            else:
                print("No symmetry found!")
                if write_cif:
                    CifWriter(symmetrized_average_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                return symmetrized_average_structure

    def get_average_structure(self, structure_unrelax: Structure, sm: Tuple[int, int, int], write_cif: bool = False):

        structure_relax = self.superlattice_relax

        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        def find_closest_string(target: str, string_list: list) -> str:
            min_distance = float('inf')
            closest_str = ""
            target = target.strip()
            for s in string_list:
                s_strip = s.strip()
                distance = levenshtein_distance(target, s_strip)
                if distance < min_distance:
                    min_distance = distance
                    closest_str = s
            return closest_str

        def match_atom_between_relax(pre_coord, post_coord, post_idx, tol=0.5):
            pre = np.array(pre_coord, dtype=float)
            post = np.array(post_coord, dtype=float)
            delta = post - pre
            delta_normalized = delta - np.round(delta)
            matched_post = pre + delta_normalized
            displacement_norm = np.linalg.norm(delta_normalized)
            is_same_atom = displacement_norm < tol
            return is_same_atom, matched_post, displacement_norm, post_idx
        
        def constrain_coords_batch(coords):
            coords_np = np.array(coords)
            mask_gt = coords_np > 0.5
            mask_lt = coords_np < -0.5
            coords_np[mask_gt] -= 1
            coords_np[mask_lt] += 1
            return coords_np.tolist() if isinstance(coords, list) else coords_np

        # Element parsingFunction
        def parse_partial_species(species_str):
            pattern = r'([A-Za-z]+)|(\d+\.?\d*)|([\+\-])'
            a = str()
            if ", " in species_str:
                species_str1 = species_str.split(", ")
            elif "," in species_str:
                species_str1 = species_str.split(",")
            else:
                species_str1 = [species_str]
            for s in species_str1:
                if ":" in s:
                    s_number = s.split(":")
                    matches = re.findall(pattern, s_number[0])
                    separated = [group for match in matches for group in match if group]
                    a = a + f"{separated[0]}{s_number[1]}"
                else:
                    matches = re.findall(pattern, s)
                    separated = [group for match in matches for group in match if group]
                    a = separated[0]
            return Composition(a)

        scale_x = sm[0]
        scale_y = sm[1]
        scale_z = sm[2]
        prim_a = 1 / scale_x
        prim_b = 1 / scale_y
        prim_c = 1 / scale_z
        trans_vectors = []
        for i in range(scale_x):
            for j in range(scale_y):
                for k in range(scale_z):
                    trans_vec = np.array([i * prim_a, j * prim_b, k * prim_c])
                    trans_vectors.append(trans_vec)

        relax_sites_by_elem = {}
        # average_structure_relax = self.get_average_composition()
        for idx, site in enumerate(structure_relax):
            elem = site.species_string
            for j, composition in enumerate(self.target_composition):
                e = str(composition)[1:-1].replace("'","").replace(" 0", "0")
                if e not in relax_sites_by_elem:
                    relax_sites_by_elem[e] = []
            if elem not in relax_sites_by_elem:
                relax_sites_by_elem[elem] = []
            relax_sites_by_elem[elem].append({
                "idx": idx,
                "cart": site.coords,
                "frac": site.frac_coords
            })
            for j, composition in enumerate(self.target_composition):
                composition_list = list(composition.keys())
                if elem in composition_list:
                    e = str(composition)[1:-1].replace("'","").replace(" 0", "0")
                    relax_sites_by_elem[e].append({
                    "idx": idx,
                    "cart": site.coords,
                    "frac": site.frac_coords
                })

        result = {}
        ele = structure_unrelax[0].species_string
        fc = structure_unrelax[0].frac_coords
        element_list = list(set(list(relax_sites_by_elem.keys())))
        for i, site in enumerate(structure_unrelax):
            result[i] = []
            for vector in trans_vectors:
                ele = site.species_string
                ele = find_closest_string(ele, element_list)
                # -------------------------Temporary patch---------------------------
                if ele == 'Nb:0.938, Sb:0.0625':
                    ele = 'Nb:0.9375, Sb:0.0625'
                # -------------------------Temporary patch---------------------------
                fc = site.frac_coords / scale_x
                fc_trans = np.array([fc[0] + vector[0], fc[1] + vector[1], fc[2] + vector[2]])
                relax_candidates = relax_sites_by_elem.get(ele, [])
                candidate_atoms = None
                min_dist = float("inf")
                for candidate in relax_candidates:
                    mabr = match_atom_between_relax(fc_trans, candidate["frac"], candidate["idx"])
                    if mabr[0] == True:
                        if mabr[2] < min_dist:
                            min_dist = mabr[2]
                            candidate_atoms = mabr[3]
                result[i].append([candidate_atoms, vector])
        frac_result = []
        for i in result:
            frac = np.zeros((1, 3))
            fracs_relax = result[i]
            for j in fracs_relax:
                carx =  structure_relax[j[0]].frac_coords - j[1]
                fr = constrain_coords_batch(carx)
                fr = fr.flatten() if isinstance(fr, np.ndarray) else np.array(fr)
                frac = frac + fr
            frac = frac / (scale_x*scale_y*scale_z) * scale_x
            frac_result.append(frac)

        relax_super_lattice = structure_relax.lattice
        base_params = {
            'a': relax_super_lattice.a / sm[0],
            'b': relax_super_lattice.b / sm[1],
            'c': relax_super_lattice.c / sm[2],
            'alpha': relax_super_lattice.alpha, 
            'beta': relax_super_lattice.beta,
            'gamma': relax_super_lattice.gamma
        }
        lattice = Lattice.from_parameters(**base_params)

        structure = Structure(
                    lattice=lattice,
                    species=[],
                    coords=[],
                    to_unit_cell=True,
                )

        for i, site in enumerate(structure_unrelax):
            frac_coord = frac_result[i]
            atom = site.species_string
            if len(atom) > 4:
                # -------------------------Temporary patch---------------------------
                if atom == 'Nb:0.938, Sb:0.0625':
                    atom = 'Nb:0.9375, Sb:0.0625'
                if atom == 'Nb:0.0625, Sb:0.938':
                    atom = 'Nb:0.0625, Sb:0.9375'
                # -------------------------Temporary patch---------------------------
                ele = parse_partial_species(atom)
            else:
                ele = atom
            frac_coord = frac_coord.tolist() if isinstance(frac_coord, np.ndarray) else frac_coord
            structure.append(
                    species = ele,
                    coords = frac_coord[0],
                    properties={'occupancy': 1})
        if write_cif:
            CifWriter(structure).write_file(os.path.join(self.output, "average_structure.cif")) 
        return structure
    
class find_refined_structure:
    def __init__(
            self,
            lattice_relax: Structure = None,
            target_sp: int = None,
            output: str = current_dir
    ):
        
        self.lattice_relax = lattice_relax
        self.output = output
        self.target_sp = target_sp
        # EnsureOutputDirectoryExists
        os.makedirs(self.output, exist_ok=True)

    def get_symmetrized_structure(self, structure, symprec):
        analyzer = SpacegroupAnalyzer(structure=structure, symprec=symprec, angle_tolerance=10)
        sg = analyzer.get_space_group_number()
        symmetrized_structure = analyzer.get_symmetrized_structure()
        return symmetrized_structure, sg
        
    def get_refined_structure_from_cif(self, structure, symprec):
        analyzer = SpacegroupAnalyzer(structure=structure, symprec=symprec)
        symmetrized_structure = analyzer.get_refined_structure()
        return symmetrized_structure
    
    def get_refined_structure(self, write_cif: bool = False):

        k = 0
        symprec_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
        best_symmetry = 0
        best_symprec = 0
        for i in symprec_list:
            k += 1
            symmetrized_average_structure, sgn = self.get_symmetrized_structure(self.lattice_relax, i)      
            if sgn > 1:
                best_symmetry = sgn
                best_symprec = i
                if best_symmetry == self.target_sp:
                    print(f"Found target symmetry! Space group={best_symmetry}, symprec={best_symprec}")
                    refined_structure = self.get_refined_structure_from_cif(symmetrized_average_structure, best_symprec)
                    if write_cif:
                        CifWriter(refined_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                    return refined_structure
        if k == len(symprec_list):
            if best_symmetry != self.target_sp and best_symmetry != 0:
                print(f"Target symmetry not found!Best space group={best_symmetry}, Bestsymprec={best_symprec}")
                refined_structure = self.get_refined_structure_from_cif(symmetrized_average_structure, best_symprec)
                if write_cif:
                    CifWriter(refined_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                return refined_structure
            else:
                print("No symmetry found!")
                if write_cif:
                    CifWriter(symmetrized_average_structure).write_file(os.path.join(self.output, "refined_structure.cif"))
                return symmetrized_average_structure