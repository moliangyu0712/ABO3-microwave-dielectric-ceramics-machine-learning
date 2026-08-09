from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.ion import Ion
from pymatgen.core.structure import Structure

import pandas as pd
import math
import traceback
import math
import os

from pymatgen.core.sites import Site
from features.composition_fixed import fixed_species_string
Site.species_string = fixed_species_string

from features.element_info import split_ion_to_element

def get_ideal_valences(structure: Structure) -> list:
    """Get ideal valence state for each element in structure"""
    BVA = BVAnalyzer()
    valence_list = BVA.get_valences(structure)
    result = []
    for i in valence_list:
        if type(i) is not list:
            i = [i]
            result.append(i)
        else:
            result.append(i) 
    """
    ideal_valences = {}
    for i, site in enumerate(structure):
        element = site.species_string
        valence = valence_list[i]
        ideal_valences[element] = valence
    """
    
    return result

def get_bond_valences(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_bond_info
    from features.env import get_dope_index_info
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    bond_valence = pd.read_csv(f"{current_dir_path}/datas/Bond_Valence.csv")
    # Get bond valence / actual valence state for each bond in structure
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    bv_info = pd.DataFrame()
    for n, row in bond_info.iterrows():
        src = row['src']
        dst = row['dst']
        dist = row['distance']
        valence = row['valence_src']
        if valence > 0:
            for i, bond_row in bond_valence.iterrows():
                    if (bond_row['Cation'] == src and 'O' == dst and bond_row['valence'] == valence) or \
                        (bond_row['Cation'] == src and 'Cl' == dst and bond_row['valence'] == valence) or \
                        (bond_row['Cation'] == src and 'F' == dst and bond_row['valence'] == valence):
                        Rij = bond_row[dst]
                        bv = math.exp((Rij - dist) / 0.37)
                        bv_info = bv_info._append({"src_index":row['src_index'],
                                                    "src":src,
                                                    "src_ratio":row['src_ratio'],
                                                    "dst":dst,
                                                    "distance":dist,
                                                    "bond_valence":bv}, ignore_index=True)
                    if (bond_row['Cation'] == src and 'O' == dst and bond_row['valence'] == valence) or \
                        (bond_row['Cation'] == src and 'Cl' == dst and bond_row['valence'] == valence) or \
                        (bond_row['Cation'] == src and 'F' == dst and bond_row['valence'] == valence):
                        Rij = bond_row[dst]
                        bv = math.exp((Rij - dist) / 0.37)
                        bv_info = bv_info._append({"src_index":row['dst_index'],
                                                    "src":dst,
                                                    "src_ratio":row['src_ratio'],
                                                    "dst":src,
                                                    "distance":dist,
                                                    "bond_valence":0-bv}, ignore_index=True)
    
    index_info = get_dope_index_info(structure)
    bv_result = pd.DataFrame()
    bv_result["src_index"] = bv_info["src_index"]
    bv_result["src"] = bv_info["src"]
    bv_result["dst"] = bv_info["dst"]
    bv_result["ratio"] = bv_info["src_ratio"]
    bv_result["bv"] = bv_info.groupby(["src_index", "src", "dst"])["bond_valence"].transform("sum")
    bv_result = bv_result.drop_duplicates().reset_index(drop=True)
    bv_result = bv_result.sort_values(by="src_index").reset_index(drop=True)
    retuslt1 = pd.DataFrame()
    for key in index_info:
        index_list = index_info[key]
        for j in index_list:
            index = key
            bv = bv_result.loc[bv_result['src_index'] == j, 'bv'].values[0]
            ratio = bv_result.loc[bv_result['src_index'] == j, 'ratio'].values[0]
            src = bv_result.loc[bv_result['src_index'] == j, 'src'].values[0]
            dst = bv_result.loc[bv_result['src_index'] == j, 'dst'].values[0]
            retuslt1 = retuslt1._append({"src_index": index,
                                            "src": src,
                                            "dst": dst,
                                            "ratio": ratio,
                                            "bv": bv}, ignore_index=True)

    ele_bv = {}
    for n, row in bv_info.iterrows():
        if row["src_index"] not in ele_bv:
            ele_bv[row["src_index"]] = row["bond_valence"] * row["src_ratio"]
        else:
            ele_bv[row["src_index"]] += row["bond_valence"] * row["src_ratio"]
    
    result = {}
    for key in index_info:
        if len(index_info[key]) > 1:
            total_bv = 0
            for idx in index_info[key]:
                total_bv += ele_bv.get(idx, 0)
        else:
            total_bv = ele_bv.get(index_info[key][0], 0)
        result[key] = total_bv

    return result, retuslt1

def get_bvsum(structure: Structure, formula = None, spn = None) -> dict:
    """Get cation and anion bond valence sum in structure"""
    bvs, _ = get_bond_valences(structure, formula, spn)
    bvsc = 0
    bvsa = 0
    nc = 0
    na = 0
    for key in bvs:
        if bvs[key] > 0:
            bvsc += bvs[key]
            nc += 1
        else:
            bvsa += bvs[key]
            na += 1
    bvsc = bvsc / nc if nc > 0 else 0
    bvsa = bvsa / na if na > 0 else 0
    return bvsc, bvsa

def get_valence_logical(structure: Structure):
    """
    Fixed version: valence calculation function adapted for mixed occupancy
    Core fixes: 
    1. Compatible with mixed occupancy formula parsing (replaces Ion.from_formula)
    2. Float precision tolerance relaxed to ±0.05
    3. Simplified dual-variable valence loop logic, fixed break level issue
    4. Preserve all elements in mixed occupancy sites, fixing NaN issues for Nb etc.
    """
    # Import external dependencies (ensure features.env module exists)
    from features.env import get_dope_index_info
    from features.env import get_dope_structure
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    # Read valence table
    file = pd.read_csv(f'{current_dir_path}/datas/valence.csv')
    valence_search = file.drop(['Atomic_Number'],axis=1)
    # Initialize variables
    valence_list = pd.DataFrame()
    single = 0  # Single-valence element valence index
    valence_sum = 0  # Final total valence sum (target ≈0)
    valence_calculation = 0  # Real-time calculated valence sum
    need_del = []  # Elements with determined valence (to be removed later)

    # -------------------------- Fixed1: Structure preprocessing for mixed occupancy compatibility --------------------------
    formula = structure.composition.reduced_formula
    sym_finder = SpacegroupAnalyzer(structure)
    structure = sym_finder.get_conventional_standard_structure()  # Standardize structure

    # Manually parse mixed occupancy element composition (replaces original Ion.from_formula, compatible with decimal occupancy ratios)
    comp_dict = {}
    for site in structure:
        for species, ratio in site.species.items():
            ele = str(species)  # Extract pure element symbol (e.g., Nb, Zn)
            comp_dict[ele] = comp_dict.get(ele, 0) + ratio  # Accumulate total occupancy ratio

    # -------------------------- Step1: CalculateNotVariable valenceElementCombined valence --------------------------
    for ion, count in comp_dict.items():
        # GetThisElementAllCanPossible valence states
        valence_only_one_judge = valence_search.loc[
            valence_search['element'] == ion, 'valence_state'
        ].tolist()
        # Only1CanPossible valence states = NotVariable valenceElement
        if len(valence_only_one_judge) == 1:
            valence_one = valence_only_one_judge[0]
            valence = count * valence_one  # TotalCombined valence(AtomCount×Single valence state)
            valence_calculation += valence
            valence_sum = valence_calculation
            # RecordValence state
            valence_list = valence_list._append(
                {'ion': ion, 'value_result': valence_one}, ignore_index=True
            )
            need_del.append(ion)  # Mark as doneProcessing

    # DeleteAlreadyDetermineValence stateElement
    for ion in need_del:
        if ion in comp_dict:
            del comp_dict[ion]
    # Preserve4PositionDecimal,ReduceAccuracyError
    valence_calculation = round(valence_calculation, 4)
    valence_sum = round(valence_sum, 4)
    # -------------------------- Step2: ProcessingSingle variable valenceElement(Remaining1) --------------------------
    if len(comp_dict) == 1:
        ion, count = list(comp_dict.items())[0]
        valence_mul_judge = valence_search.loc[
            valence_search['element'] == ion, 'valence_state'
        ].tolist()
        found = False
        for valence_ion in valence_mul_judge:
            temp_calc = valence_calculation + count * valence_ion
            # Fixed2: FloatCountAccuracyDetermine(±0.05)
            if abs(temp_calc) < 0.05:
                valence_sum = temp_calc
                valence_list = valence_list._append(
                    {'ion': ion, 'value_result': valence_ion}, ignore_index=True
                )
                found = True
                break
            single += 1
        if not found:
            raise ValueError(f"Single variable valenceElement{ion}NotFoundFullSufficientCombine valence balanced state(±0.05Error)")

    # -------------------------- Step3: ProcessingDual-valence element(Remaining2) --------------------------
    elif len(comp_dict) == 2:
        ions = list(comp_dict.keys())
        ion1, count1 = ions[0], comp_dict[ions[0]]
        ion2, count2 = ions[1], comp_dict[ions[1]]
        # Getof two elementsAllCanPossible valence states
        val1_list = valence_search.loc[valence_search['element'] == ion1, 'valence_state'].tolist()
        val2_list = valence_search.loc[valence_search['element'] == ion2, 'valence_state'].tolist()
        found = False

        # Simplify dual valenceLoop,AvoidHierarchy confusion
        for val1 in val1_list:
            for val2 in val2_list:
                temp_calc = valence_calculation + count1 * val1 + count2 * val2
                # Fixed2: FloatCountAccuracyDetermine(±0.05)
                if abs(temp_calc) < 0.05:
                    valence_sum = temp_calc
                    # Record valence states of two elements
                    valence_list = valence_list._append(
                        {'ion': ion1, 'value_result': val1}, ignore_index=True
                    )
                    valence_list = valence_list._append(
                        {'ion': ion2, 'value_result': val2}, ignore_index=True
                    )
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Dual-valence element{ion1}/{ion2}NotFoundFullSufficientCombine valence balanced state(±0.05Error)")

    # -------------------------- Step4: ProcessingMulti variable valenceElement(>2) --------------------------
    else:
        raise ValueError(f'Multi variable valenceElementCount={len(comp_dict)},ExceedTwo,NoneMethodCalculateCombined valence')

    # -------------------------- Fixed3: MatchstructuresSite(Preserve mixed occupancyAll elements) --------------------------
    structure1 = get_dope_structure(structure)
    element_list = pd.DataFrame()
    # Traverse sites,Preserve mixed occupancyAll elements(Such asNb0.67Zn0.33SiteSimultaneouslyRecordNbandZn)
    for i, ele in enumerate(structure1):
        for species, _ in ele.species.items():
            ion = str(species)
            element_list = element_list._append(
                {'index': i, 'ion': ion}, ignore_index=True
            )

    # Valence stateMatchToAtomSite
    valence_list = valence_list.merge(
        element_list, on='ion', how='right'
    ).sort_values(by='index').reset_index(drop=True)

    # -------------------------- ByDopingIndexOrganizeFinalValence state --------------------------
    dope_index = get_dope_index_info(structure)
    valences = []
    for key in dope_index:
        index_list = dope_index[key]
        valence = []
        for j in index_list:
            # ExtractThisSiteAllValence state(Deduplicate+Removenan)
            vals = valence_list.loc[
                valence_list['index'] == j, 'value_result'
            ].dropna().unique().tolist()
            valence.extend(vals)
        valences.append(valence)
    return valences

def get_valence_reference(structure: Structure) -> pd.DataFrame:
    """Getstructureseach inBond valence referenceValue"""
    from features.env import get_dope_index_info
    from features.env import get_dope_structure
    element = {"Li":1, "O":-2, "Na":1, "Mg":2, "Al":3, 
               "Ca":2, "Ti":4, "V":5, "Cr":3, "Mn":4, "Fe":3, 
               "Co":2, "Ni":2, "Zn":2, "Ga":3, "Sr":2, 
               "Y":3, "Zr":4, "Nb":5, "Cd":2, "In":3, "Sn":4, 
               "Sb":5, "Ba":2, "La":3, "Ce":3, "Pr":3, 
               "Nd":3, "Sm":3, "Eu":3, "Gd":3, "Tb":3, 
               "Dy":3, "Ho":3, "Er":3, "Tm":3, "Yb":3, 
               "Lu":3, "Hf":4, "Ta":5, "W":6, "Pb":2, "Bi":3}
    structure1 = get_dope_structure(structure)
    valence_list = pd.DataFrame()
    for i, ele in enumerate(structure1):
        ion = str(ele.species_string)
        valence_value = element.get(ion, 0)
        valence_list = valence_list._append({'index':i, 'value':valence_value}, ignore_index=True)
    dope_index = get_dope_index_info(structure)
    valences = []
    for key in dope_index:
        index_list = dope_index[key]
        valence = []
        if len(index_list) > 1:
            for j in index_list:
                valence.append(valence_list.loc[valence_list['index'] == j, 'value'].values[0])
        else:
            valence.append(valence_list.loc[valence_list['index'] == index_list[0], 'value'].values[0])
        valences.append(valence)
    return valences

def get_valence(structure: Structure):
    """
    Valence with fallback strategyGetMainFunction(CoreFunction)
    Priority: get_valence_reference > get_ideal_valences > get_valence_logical
    
    Args:
        structure: pymatgenStructureCorrectObject,ReplaceTableCrystal structure
    
    Returns:
        List[List[int | float]]: Valence list for each element(andOriginalFunctionReturnsFormatOneLead to)
    
    Raises:
        Exception: Raise when all three valence acquisition functions fail,IncludeAllFailedReason
    """
    # StoreEachFunctionFailedInformation,UseAtFinalReportWrong
    error_infos = []
    # OneStep: Try calling get_valence_reference(TimePriority)
    try:
        valences = get_valence_reference(structure)
        return valences
    except Exception as e:
        # RecordFailedReason(IncludeStackInformation,For easier diagnosis)
        error_info = f"get_ideal_valences Failed: {str(e)}\n{traceback.format_exc()}"
        error_infos.append(error_info)
    # TwoStep: Try calling get_ideal_valences(MostHighPriority)
    try:
        valences = get_ideal_valences(structure)
        return valences
    except Exception as e:
        error_info = f"get_valence_logical Failed: {str(e)}\n{traceback.format_exc()}"
        error_infos.append(error_info)
    # ThreeStep: Try calling get_valence_logical(MostLowPriority)
    try:
        valences = get_valence_logical(structure)
        return valences
    except Exception as e:
        error_info = f"get_valence_reference Failed: {str(e)}\n{traceback.format_exc()}"
        error_infos.append(error_info)

    # AllFunctionAverageFailed,ThrowOutputCollectTotalException
    raise Exception(
        "AllValence stateGetMethodAverageFailed!\n" + "\n------------------------\n".join(error_infos)
    )