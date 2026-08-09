from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
import pandas as pd
import numpy as np
import re
from pymatgen.core import Structure, Lattice, Composition
from features.valence import get_valence
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer #Get space group information
import os
current_dir_path = os.path.dirname(os.path.abspath(__file__))
cutoff_list = pd.read_csv(f"{current_dir_path}/datas/VESTA_Cutoffs.csv")
cn_list = pd.read_csv(f"{current_dir_path}/datas/cn_list.csv")

from pymatgen.core.sites import Site
from features.composition_fixed import fixed_species_string
Site.species_string = fixed_species_string

import numpy as np

# Utility function: generate {multiplicity+symbol: [coordinate list]} dictionary
def get_wyckoff_coord_dict(symmetry_equivalent_wyckoff_positions):
    result = {}
    for wyckoff in symmetry_equivalent_wyckoff_positions:
        element = extract_elements(wyckoff['species'])[0]
        multiplicity = wyckoff['multiplicity']
        wyckoff_symbol = wyckoff['wyckoff_symbol']
        key = f"{element}_{multiplicity}{wyckoff_symbol}"
        result[key] = wyckoff['all_equivalent_coords']
    return result

# Utility function: match Wyckoff symbol by coordinates (e.g., 4b, 16f)
def get_wyckoff_by_coord(coord, wyckoff_dict, tol=1e-4):
    for wyckoff_key, coord_list in wyckoff_dict.items():
        for wc in coord_list:
            if np.allclose(coord, wc, atol=tol):
                return wyckoff_key
    return "unknown"


def extract_elements(ion_string):
    """
    Extract element symbols from ion string
    
    Parameters:
        ion_string: String containing ion representation,For example "Ba2+,Na+,I-"
    
    Returns:
        List of element symbols,For example ['Ba', 'Na', 'I']
    """
    # Use regex to match element symbols
    # Element symbol consists of 1-2 letters, first uppercase, second lowercase (if present)
    pattern = r'([A-Z][a-z]?)'
    
    # Find all matching element symbols
    elements = re.findall(pattern, ion_string)
    
    return elements

def _get_lattice_parameters(sg, struc_parameters):
    """Get optimized lattice parameters based on space group"""
    if sg in [0, 1, 2]: #Triclinic
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['b'], 
            'c': struc_parameters['c'], 
            'alpha': struc_parameters['alpha'],
            'beta': struc_parameters['beta'],
            'gamma': struc_parameters['gamma']
        }
    elif sg in list(range(3, 15 + 1)): #Monoclinic
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['b'], 
            'c': struc_parameters['c'], 
            'alpha': 90,
            'beta': struc_parameters['beta'],
            'gamma': 90
        }
    elif sg in list(range(16, 74 + 1)): #Orthorhombic
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['b'], 
            'c': struc_parameters['c'], 
            'alpha': 90,
            'beta': 90,
            'gamma': 90
        }
    elif sg in list(range(75, 142 + 1)): #Tetragonal
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['a'], 
            'c': struc_parameters['c'], 
            'alpha': 90,
            'beta': 90,
            'gamma': 90
        }
    elif sg in list(range(143, 194 + 1)): #Trigonal/Hexagonal
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['a'], 
            'c': struc_parameters['c'], 
            'alpha': 90,
            'beta': 90,
            'gamma': 120
        }
    elif sg in list(range(195, 230 + 1)): #Cubic
        base_params = {
            'a': struc_parameters['a'],
            'b': struc_parameters['a'], 
            'c': struc_parameters['a'], 
            'alpha': 90,
            'beta': 90,
            'gamma': 90
        }
    return base_params

def get_dope_coords(structure: Structure) -> np.ndarray:
    coords_list = []
    for site in structure:
        element_str = site.species_string
        if len(element_str) <= 4: # undope
            coords = site.coords
            coords_list.append(coords)
        else:
            elements = element_str.split(",")
            for e in elements:
                coords = site.coords
                # coords = site.coords + np.random.normal(0, 1e-6, size=3) # Small perturbations are added to the coordinates of the doped atoms (1e-6 Å magnitude)
                coords_list.append(coords)
    return np.array(coords_list)

def get_dope_structure(structure:Structure):
    space_group_number = structure.get_space_group_info()[1]
    space_group = structure.get_space_group_info()[0]
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha = structure.lattice.alpha
    beta = structure.lattice.beta
    gamma = structure.lattice.gamma
    struc_parameters = {"a":a, "b":b, "c":c, "alpha":alpha, "beta":beta, "gamma":gamma}
    lattice_params = _get_lattice_parameters(space_group_number, struc_parameters)
    lattice = Lattice.from_parameters(**lattice_params)
    structure_doped = Structure.from_spacegroup(
        space_group,
        lattice = lattice,
        species=[],
        coords=[]
        )

    sga = SpacegroupAnalyzer(structure)
    sym_struct = sga.get_symmetrized_structure()
    equiv_indices = sym_struct.equivalent_indices
    index_to_origin = {}
    for base_idx, equivs in enumerate(equiv_indices):
        for idx in equivs:
            index_to_origin[idx] = base_idx

    for i, ele in enumerate(structure):
        frac_coord = ele.frac_coords
        atom = ele.species_string
        origin_idx = index_to_origin[i]  # Original symmetry index

        if len(str(ele.species_string)) > 4: # Doping site
            elements = atom.split(",")
            for element in elements:
                e_r_list = element.split(":")
                structure_doped.append(
                    species = Composition(e_r_list[0]),
                    coords = frac_coord,
                    properties={
                        'occupancy': float(e_r_list[1]),
                        'origin_index': origin_idx  # [Save]
                                }
                )
        else: # Non-doping site
            e = ele
            structure_doped.append(
                species = Composition(extract_elements(ele.species_string)[0]),
                coords = frac_coord,
                properties={
                    'occupancy': 1,
                    'origin_index': origin_idx  # [Save]
                    })
    return structure_doped

def get_cationic_coordination_numbers(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    """Get coordination number for each atom in structure"""
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    bond_info["coordination_number"] = bond_info.groupby('src_index')['src'].transform('count')
    bond_info1 = bond_info.drop_duplicates(subset=['src_index'])
    coordination_number = bond_info1[["src_index", "src", "src_ratio",  "valence_src", "coordination_number"]]
    coordination_number1 = coordination_number.sort_values(by="src_index", ascending=True).reset_index(drop=True)
    return coordination_number1

def get_anionic_coordination_numbers(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    """Future implementation: get coordination number for each anion"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure) 
    bond_info = get_bond_info(structure, dope = False, formula = formula, spn = spn)
    bond_info["coordination_number"] = bond_info.groupby('dst_index')['dst'].transform('count')
    bond_info1 = bond_info.drop_duplicates(subset=['dst_index'])
    coordination_number = bond_info1[["dst_index", "dst", "dst_ratio",  "valence_dst", "coordination_number"]]
    anionic_index_dope = []
    for index in coordination_number["dst_index"].values.tolist():
        index_dope = index_info[index][0]
        anionic_index_dope.append(index_dope)
    coordination_number["dst_index"] = anionic_index_dope
    coordination_number1 = coordination_number.sort_values(by="dst_index", ascending=True).reset_index(drop=True)
    return coordination_number1

def get_dope_index_info(structure: Structure) -> dict:
    """GetstructuresInDoping siteIndexInformation"""
    dope_index = {}
    k = 0
    for i, ele in enumerate(structure):
        dope_index[i] = []
        if len(str(ele.species_string)) > 4: # Doping site
            elements = ele.species_string.split(",")
            for element in elements:
                dope_index[i].append(k)
                k += 1
        else: # Non-doping site
            dope_index[i].append(k)
            k += 1
    return dope_index

# Note: ThisPartialOnlySupportABO3Perovskite material,If neededUsingThisPartialProceedOtherstructuresCalculateNeedTo entercn_listCustomAndModifyRelatedCode
def get_cn_with_list(formula, bond_list, structure: Structure, spn = None):
    """
    formula: Ba3MgTa2O9, BaTiO3etc.
    """
    # ExtractGetBracketInsideInsidecontent and non-bracket elements,Adapt to multiple scenarios(FixedElementSplitIssue)
    def separate_brackets(text):
        stack = []
        result = []
        current = ""

        for char in text:
            if char == "(":
                # left parenthesis encountered,FirstSplitCurrentaccumulated non-bracket contentInsideContent(SplitAsSingleElement),Then addResult
                if current:
                    split_elements(current, result)
                    current = ""  # ResetCurrentInsideContent,Prepare to receive content inside brackets
                stack.append("(")  # Stack mark left bracket,Used to determine bracket closure
            elif char == ")":
                # EncounterToRight bracket,Directly add accumulated bracket content to result(BracketInsideAsCombinationElement,NotSplit)
                if current:
                    result.append(current)
                    current = ""  # ResetCurrentInsideContent,StandardPrepareReceiveBackContinueNon-bracketInsideContent
                if stack:
                    stack.pop()  # Close bracket,PopOutputStackInLeft bracket marker
            else:
                # Non-bracket character,AccumulateToCurrentInsideContentIn(IncludeElementSymbol, Number, DecimalPoint)
                current += char
        
        # LoopEndBack,SplitRemainingNotadded non-bracket contentInsideContent,AddEnterResult
        if current:
            split_elements(current, result)
        
        return result

    def split_elements(content, result):
        """AuxiliaryFunction: SplitNoneBracketElementInsideContent(Such asLaAlO3SplitAsLa, Al, O3)"""
        i = 0
        n = len(content)
        while i < n:
            # ElementSymbol: First characterLargeWrite,BackContinueCanFollowSmallwrite letters(Such asLa, Al)
            if content[i].isupper():
                element = content[i]
                i += 1
                # MatchBackcontinuousSmallwrite letters(Such asTi, Sr)
                while i < n and content[i].islower():
                    element += content[i]
                    i += 1
                # MatchBackcontinuousNumberandDecimalPoint(Such asO3, Al0.8)
                while i < n and (content[i].isdigit() or content[i] == "."):
                    element += content[i]
                    i += 1
                result.append(element)
            else:
                # ExceptionSituation(Such asPureNumber,Directly add here,Adjustable based on requirements)
                result.append(content[i])
                i += 1
    if spn:
        space_group_number = spn
    else:
        sym_finder = SpacegroupAnalyzer(structure)
        structure = sym_finder.get_conventional_standard_structure()
        space_group_number = sym_finder.get_space_group_number() #Space group number
    element_number = separate_brackets(formula)
    A_site = element_number[0]
    On = element_number[-1]
    A_cn = 0
    B_cn = 0
    for row in cn_list.values.tolist():
        f, sp, As, Bs = row
        if int(sp) == space_group_number:
            if On in f:
                A_cn = As
                B_cn = Bs
    bond_list = bond_list.sort_values(by=['src_index', 'distance'], ascending=[True, True]).reset_index(drop=True)
    rule = {}
    for row in bond_list.values.tolist():
        src_index, src, src_ratio, valence_src, src_wyckoff, src_origin_index, dst_index, dst, dst_ratio, valence_dst, dst_wyckoff, dst_origin_index, distance = row
        value = (src_index, src)
        if value not in rule:
            if src in A_site:
                rule[value] = A_cn
            else:
                if src != 'O':
                    rule[value] = B_cn
    result_list = []
    # TraverseEachA+BCombinationPreserveRuleThen
    for (src_index, src), cn in rule.items():
        # Filter coordination numbers
        group_data = bond_list[(bond_list['src_index'] == src_index) & (bond_list['src'] == src) & (bond_list['dst'] == 'O')]
        # GetFrontkeep_numRow
        cn_data = group_data.head(cn)
        result_list.append(cn_data)
    return result_list

def get_bond_info(structure: Structure, dope: bool, formula = None, spn = None) -> pd.DataFrame:
    numerical_tol = 1.0e-8
    pbc = np.array([1, 1, 1], dtype=np.int64)
    valence = get_valence(structure)
    cart_coords = structure.cart_coords

    from GNBO.crystal.Crystal_information import get_symmetry_equivalent_wyckoff_positions
    sym_wyckoff = get_symmetry_equivalent_wyckoff_positions(structure)
    wyckoff_dict = get_wyckoff_coord_dict(sym_wyckoff)
    structure_used = structure
    if dope:
        structure_used = get_dope_structure(structure)
        cart_coords = get_dope_coords(structure_used)
    lattice_matrix = structure_used.lattice.matrix
    src_id, dst_id, images, bond_dist = find_points_in_spheres(
        cart_coords, cart_coords, r=5, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )

    exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
    src_id, dst_id, images, bond_dist = (
        src_id[exclude_self], dst_id[exclude_self], images[exclude_self], bond_dist[exclude_self]
    )

    if len(cart_coords) > len(structure):
        mask = ~(np.all(np.abs(cart_coords[src_id] - cart_coords[dst_id]) == 0, axis=1) & (src_id != dst_id))
        src_id, dst_id, images, bond_dist = src_id[mask], dst_id[mask], images[mask], bond_dist[mask]

    valence_list = []
    if dope:
        for i in valence:
            if isinstance(i, list):
                valence_list.extend(i)
            else:
                valence_list.append(i)
    else:
        for i in valence:
            valence_list.append(i[0])

    bil = pd.DataFrame()
    for n, src_index in enumerate(src_id.tolist()):
        src = extract_elements(str(structure_used[src_index].species))[0]
        src_r = structure_used[src_index].properties.get('occupancy', 1)
        src_origin = structure_used[src_index].properties.get('origin_index',100)  # <-- DirectlyRead
        src_coord = structure_used[src_index].frac_coords
        src_wyckoff = get_wyckoff_by_coord(src_coord, wyckoff_dict)
        src_wyckoff = src_wyckoff.split("_")[-1] if src_wyckoff != "unknown" else "unknown"

        dst_index = dst_id.tolist()[n]
        dst = extract_elements(str(structure_used[dst_index].species))[0]
        dst_r = structure_used[dst_index].properties.get('occupancy', 1)
        dst_origin = structure_used[dst_index].properties.get('origin_index',100)  # <-- DirectlyRead
        dst_coord = structure_used[dst_index].frac_coords
        dst_wyckoff = get_wyckoff_by_coord(dst_coord, wyckoff_dict)
        dst_wyckoff = dst_wyckoff.split("_")[-1] if dst_wyckoff != "unknown" else "unknown"

        valence_src = valence_list[src_index]
        valence_dst = valence_list[dst_index]

        bil = pd.concat([bil, pd.DataFrame([{
            "src_index": src_index,
            "src": src,
            "src_ratio": src_r,
            "valence_src": valence_src,
            "src_wyckoff": src_wyckoff,
            "src_origin_index": src_origin,    
            "dst_index": dst_index,
            "dst": dst,
            "dst_ratio": dst_r,
            "valence_dst": valence_dst,
            "dst_wyckoff": dst_wyckoff,
            "dst_origin_index": dst_origin,    
            "distance": bond_dist.tolist()[n]
        }])], ignore_index=True)

    result = pd.DataFrame()
    if formula:
        cn_list_out = get_cn_with_list(formula, bil, structure_used, spn)
        if cn_list_out:
            result = pd.concat(cn_list_out).reset_index(drop=True)
    else:
        for row in bil.values.tolist():
            src_index, src, src_ratio, valence_src, src_wyckoff, src_origin, \
            dst_index, dst, dst_ratio, valence_dst, dst_wyckoff, dst_origin, distance = row

            cutoff = cutoff_list[(cutoff_list["element1"] == src) & (cutoff_list["element2"] == dst)]["cutoff"]
            if len(cutoff.values) > 0 and float(distance) <= cutoff.values[0]:
                result = pd.concat([result, pd.DataFrame([{
                    "src_index": src_index,
                    "src": src,
                    "src_ratio": src_ratio,
                    "valence_src": valence_src,
                    "src_wyckoff": src_wyckoff,
                    "src_origin_index": src_origin,
                    "dst_index": dst_index,
                    "dst": dst,
                    "dst_ratio": dst_ratio,
                    "valence_dst": valence_dst,
                    "dst_wyckoff": dst_wyckoff,
                    "dst_origin_index": dst_origin,
                    "distance": distance
                }])], ignore_index=True)
    return result