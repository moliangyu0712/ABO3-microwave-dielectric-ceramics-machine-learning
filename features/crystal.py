from pymatgen.symmetry.analyzer import SpacegroupAnalyzer #Get space group information
import numpy as np
from pymatgen.core import Structure
from features.valence import get_valence, get_bond_valences, get_bvsum
from pymatgen.symmetry.groups import SpaceGroup
import math

def get_GII(structure: Structure, formula = None, spn = None):
    """
    Get global instability index GII for each element
    DOI: 10.1107/S0108768107046174
    """
    ideal_valences = get_valence(structure)
    bond_valences_info = get_bond_valences(structure, formula, spn)[1]
    gii = 0
    for i, site in enumerate(structure):
        ideal_valence_list = ideal_valences[i]
        actual_valence_list = bond_valences_info.loc[bond_valences_info['src_index'] == i, 'bv'].tolist()
        ratio_list = bond_valences_info.loc[bond_valences_info['src_index'] == i, 'ratio'].tolist()
        for j, ideal_valence in enumerate(ideal_valence_list):
            gii += ((actual_valence_list[j] - ideal_valence)**2) * ratio_list[j]
    gii /= len(structure)
    gii = math.sqrt(gii)
    return gii

def analyze_crystal_data(structure: Structure, formula = None, spn = None) -> dict:
    result = {}
    if spn:
        space_group_number = spn
        crystal_system = get_crystal_system_from_spacegroup_number(spn)
    else:
        #Get space group
        sym_finder = SpacegroupAnalyzer(structure)
        structure = sym_finder.get_conventional_standard_structure()
        space_group_number = sym_finder.get_space_group_number() #Space group number
        crystal_system = sym_finder.get_crystal_system()
    #Get unit cell volume
    volume = structure.volume

    #Theoretical density
    density = structure.density 
    density = str(density)
    density = float(density.replace(" g cm^-3", ""))

    #GetGII
    GII = get_GII(structure, formula, spn)
    # Cation-anion bond valence sum
    bvsc, bvsa = get_bvsum(structure, formula, spn)

    # Lattice constant edge length standard deviation
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    le = [a, b, c]
    lsd = np.std(le)
    # Lattice constant angle standard deviation
    alpha = structure.lattice.alpha
    beta = structure.lattice.beta
    gamma = structure.lattice.gamma
    #print(f'{nam}:{alpha},{beta},{gamma}')
    an = [alpha, beta, gamma]
    asd = np.std(an)
    result['sp'] = space_group_number
    result['crystal_system'] = crystal_system
    result['density'] = density
    result['a'] = a
    result['b'] = b
    result['c'] = c
    result['lsd'] = lsd
    result['alpha'] = alpha
    result['beta'] = beta
    result['gamma'] = gamma
    result['asd'] = asd
    result['volume'] = volume
    result["GII"] = GII
    result["bvsc"] = bvsc
    result["bvsa"] = bvsa
    return result
"""
    except:
        result['sp'] = None
        result['crystal_system'] = None
        result['density'] = None
        result['a'] = None
        result['b'] = None
        result['c'] = None
        result['lsd'] = None
        result['alpha'] = None
        result['beta'] = None
        result['gamma'] = None
        result['asd'] = None
        result['volume'] = None
        result["GII"] = None
        result["bvsc"] = None
        result["bvsa"] = None
        return result
"""
    
def get_crystal_system_from_spacegroup_number(spacegroup_number):
    """
    Get corresponding crystal system based on space group number
    
    Parameters:
        spacegroup_number (int): Space group number (1-230)
    
    Returns:
        str: Crystal system name (e.g., cubic, hexagonal, tetragonal, etc.)
    """
    try:
        # Create SpaceGroup object based on space group number
        sg = SpaceGroup.from_int_number(spacegroup_number)
        
        # Get crystal system information
        crystal_system = sg.crystal_system
        
        return {
            "crystal_system": crystal_system,
        }
    
    except ValueError as e:
        print(f"Error: {e}")
        return None