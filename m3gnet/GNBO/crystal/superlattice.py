import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import random
from collections import defaultdict
from typing import Tuple

def validate_disordered_site(site):
    """Validate disordered sites: ProportionandAs1,DecimalPointPositionCount≤4"""
    total_occ = sum(site.species.values())
    if not np.isclose(total_occ, 1.0, atol=1e-6):
        raise ValueError(
            f"NoneSequenceSite {tuple(np.round(site.frac_coords, 6))} AtomProportionsumAs {total_occ:.6f},Notetc.At1!"
        )
    
    for elem, occ in site.species.items():
        occ_str = f"{occ:.10f}".rstrip('0').rstrip('.')
        if '.' in occ_str and len(occ_str.split('.')[-1]) > 4:
            raise ValueError(
                f"NoneSequenceSite {tuple(np.round(site.frac_coords, 6))} In {elem} proportion of {occ:.6f} Decimal places exceed4Position!"
            )
    return True

def find_disordered_site_features(structure):
    """CheckFindOriginalstructuresInAllNoneSequenceSite“Element combinationFeature”andProportionInformation"""
    disordered_features = {}
    for site in structure:
        if len(site.species) > 1:
            validate_disordered_site(site)
            elem_set = frozenset([elem.symbol for elem in site.species.keys()])
            species = [elem.symbol for elem in site.species.keys()]
            ratios = [occ for occ in site.species.values()]
            if elem_set not in disordered_features:
                disordered_features[elem_set] = {
                    'species': species,
                    'ratios': ratios
                }
    if not disordered_features:
        raise ValueError("Not instructuresInFoundNoneSequenceSite(Multi-element co-occupancy),NoneMethodEnterBackContinueCalculate!")
    return disordered_features

def robust_expand_structure(structure, supercell_size=(10,10,10), random_seed=None):
    """OptimizeCore: SupercellBackBy“Element combinationFeature”RecognizeNoneSequenceSite,Support multiple disordered sites"""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    orig_num_sites = len(structure)
    
    # 2. ExtractNoneSequenceSiteFeature
    disordered_features = find_disordered_site_features(structure)
    
    # 3. Supercell
    sc_x, sc_y, sc_z = supercell_size
    sc_total = sc_x * sc_y * sc_z
    expanded_structure = structure * supercell_size
    expanded_num_sites = len(expanded_structure)
    expected_total = orig_num_sites * sc_total
    if expanded_num_sites != expected_total:
        raise RuntimeError(f"SupercellFailed: Pre-Period{expected_total}Atom,Actual{expanded_num_sites}")
    
    # 4. ByFeatureClassificationSite
    feature_to_indices = defaultdict(list)
    ordered_indices = []
    for idx, site in enumerate(expanded_structure):
        current_elem_set = frozenset([elem.symbol for elem in site.species.keys()])
        if current_elem_set in disordered_features:
            feature_to_indices[current_elem_set].append(idx)
        else:
            ordered_indices.append(idx)
    
    # verificationClassificationComplete
    total_classified = sum(len(indices) for indices in feature_to_indices.values()) + len(ordered_indices)
    if total_classified != expanded_num_sites:
        raise RuntimeError(f"SiteClassificationError: ClassificationTotal count{total_classified}≠total sitesCount{expanded_num_sites}")
    
    # 5. Assign disordered site elements
    final_species = [None] * expanded_num_sites
    for feat, indices in feature_to_indices.items():
        feat_info = disordered_features[feat]
        species = feat_info['species']
        ratios = feat_info['ratios']
        feat_site_count = len(indices)
        
        # CalculateElementCountCountAndCorrectError
        elem_counts = {elem: int(round(ratio * feat_site_count)) for elem, ratio in zip(species, ratios)}
        total_count = sum(elem_counts.values())
        if total_count != feat_site_count:
            last_elem = list(elem_counts.keys())[-1]
            elem_counts[last_elem] += (feat_site_count - total_count)
        
        # GenerateAndShuffleElementColumnTable
        elem_list = []
        for elem, count in elem_counts.items():
            elem_list.extend([elem] * count)
        random.shuffle(elem_list)
        
        # AssignElement
        for idx, elem in zip(indices, elem_list):
            final_species[idx] = elem
    
    # 6. ProcessingOrdered site
    for idx in ordered_indices:
        site = expanded_structure[idx]
        final_species[idx] = list(site.species.keys())[0].symbol
    
    # 7. CreateFinalP1structures
    original_matrix = structure.lattice.matrix
    supercell_matrix = [
        original_matrix[0] * sc_x,
        original_matrix[1] * sc_y,
        original_matrix[2] * sc_z
    ]
    supercell_lattice = Lattice(supercell_matrix)
    final_coords = [expanded_structure[idx].frac_coords for idx in range(expanded_num_sites)]
    
    final_structure = Structure(
        lattice=supercell_lattice,
        species=final_species,
        coords=final_coords,
        to_unit_cell=False
    )
    
    # verificationP1Symmetry(OnlyValidate,NoneOutput)
    final_analyzer = SpacegroupAnalyzer(final_structure)
    if final_analyzer.get_space_group_number() != 1:
        raise RuntimeWarning("FinalstructuresNotReachToP1Symmetry,CanCanExistsSiteAssignException")
    
    return final_structure

def process_cif_structure(cif_file, supercell_size=(10,10,10), random_seed=None):
    """MainProcessingFunction: DirectlyReturnsFinalstructures"""
    try:
        return robust_expand_structure(cif_file, supercell_size, random_seed)
    except Exception as e:
        raise  # PreserveExceptionThrowOutput,Convenient for callerProcessing

def make_supercell(structure: Structure, scaling_matrix: Tuple[int, int, int]) -> Structure:
    """
    Crystal structurePer given supercell multiplierCountProceedSupercell
    
    Parameters:
        structure: pymatgenStructureCorrectObject
        scaling_matrix: Supercell multiplierCountTupleGroup,For example(2,2,2)TableShowInEachLattice vectorQuantityDirectional supercell2Times
    
    Returns:
        SupercellBackStructureCorrectObject
    """
    # ImportSupercellClass
    from pymatgen.core.structure import Structure as BaseStructure
    from pymatgen.transformations.standard_transformations import SupercellTransformation
    
    # CreateSupercell transformationCorrectObject
    transformation = SupercellTransformation(scaling_matrix=scaling_matrix)
    
    # CorrespondingUseSupercell transformation
    supercell_structure = transformation.apply_transformation(structure)
    
    return supercell_structure