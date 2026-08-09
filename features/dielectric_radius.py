import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.sites import Site
from features.composition_fixed import fixed_species_string
Site.species_string = fixed_species_string
from features.env import get_cationic_coordination_numbers, get_anionic_coordination_numbers
import math
import os
current_dir_path = os.path.dirname(os.path.abspath(__file__))
dielectric_data = pd.read_csv(f"{current_dir_path}/datas/Dielectric_Polarization_Extension.csv")

def get_dielectric_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    """Get dielectric constant information for structure"""
    from features.element_info import split_ion_to_element
    cationic_coordination_numbers = get_cationic_coordination_numbers(structure, formula, spn)
    anionic_coordination_numbers = get_anionic_coordination_numbers(structure, formula, spn)
    cationic_coordination_numbers = cationic_coordination_numbers.rename(columns={"src_index": "index"})
    anionic_coordination_numbers = anionic_coordination_numbers.rename(columns={"dst_index": "index"})
    cationic_coordination_numbers = cationic_coordination_numbers.rename(columns={"src": "element"})
    anionic_coordination_numbers = anionic_coordination_numbers.rename(columns={"dst": "element"})
    cationic_coordination_numbers = cationic_coordination_numbers.rename(columns={"src_ratio": "ratio"})
    anionic_coordination_numbers = anionic_coordination_numbers.rename(columns={"dst_ratio": "ratio"})
    cationic_coordination_numbers = cationic_coordination_numbers.rename(columns={"valence_src": "valence"})
    anionic_coordination_numbers = anionic_coordination_numbers.rename(columns={"valence_dst": "valence"})
    coordination_numbers = pd.concat([cationic_coordination_numbers, anionic_coordination_numbers], axis = 0, ignore_index=True)
    coordination_numbers["element"] = coordination_numbers["element"].apply(split_ion_to_element)
    dielectric_info = coordination_numbers.merge(dielectric_data, how='left', left_on=['element', 'coordination_number', 'valence'], right_on=['element', 'coordination_number', 'valence'])
    return dielectric_info
    
def get_averaged_dielectric_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    """Get average dielectric constant information for each atom in structure""" 
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)   
    elements = {}
    for i, site in enumerate(structure):
        elements[i] = site.species_string
    dielectric_info = get_dielectric_info(structure, formula, spn)
    result = pd.DataFrame()
    for i in index_info:
        index = index_info[i]
        IR = 0
        DP = 0
        VI = 0
        if len(index) > 1:
            for j in index:
                ir = dielectric_info[dielectric_info["index"] == j]["IR"].values.tolist()[0]
                vi = 4 / 3 * math.pi * (dielectric_info[dielectric_info["index"] == j]["IR"].values.tolist()[0])**3
                dp = dielectric_info[dielectric_info["index"] == j]["DP"].values.tolist()[0]
                ratio = dielectric_info[dielectric_info["index"] == j]["ratio"].values.tolist()[0]
                IR += ir * ratio
                DP += dp * ratio
                VI += vi * ratio
        else:
            ir = dielectric_info[dielectric_info["index"] == index[0]]["IR"].values.tolist()[0]
            vi = 4 / 3 * math.pi * (dielectric_info[dielectric_info["index"] == index[0]]["IR"].values.tolist()[0])**3
            dp = dielectric_info[dielectric_info["index"] == index[0]]["DP"].values.tolist()[0]
            ratio = dielectric_info[dielectric_info["index"] == index[0]]["ratio"].values.tolist()[0]
            IR += ir * ratio
            DP += dp * ratio
            VI += vi * ratio
        result = result._append({"index":i,
                                "element":elements[i],
                                "IR":IR,
                                "VI":VI,
                                "DP":DP}, ignore_index=True)
    return result