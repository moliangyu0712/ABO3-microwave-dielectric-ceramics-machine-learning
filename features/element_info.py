from pymatgen.core import Structure
import pandas as pd
import math
import os
current_dir_path = os.path.dirname(os.path.abspath(__file__))

from pymatgen.core.sites import Site
from features.composition_fixed import fixed_species_string
Site.species_string = fixed_species_string

def get_element_data(structure: Structure) -> pd.DataFrame:
    """Analyze element data from a given structure and return a DataFrame with element counts."""
    from features.valence import get_valence
    from features.env import get_dope_structure
    from features.env import get_dope_index_info
    result = pd.DataFrame()
    valences_info  = get_valence(structure)
    index_info = get_dope_index_info(structure)
    index_info = list(index_info.keys())
    dope = False
    for i, site in enumerate(structure):
        if len(str(site.species_string)) > 4: # Doping site
            dope = True
            break
    if dope:
        structure = get_dope_structure(structure)

    k = 0
    for ind in valences_info:
        for i in ind:
            ratio = structure[k].properties.get('occupancy', 1)
            result = result._append({ "index": k, "element": structure[k].species_string, 
                                     "ratio":ratio, "valence": i}, ignore_index = True)
            k += 1 
    return result

def get_mass_info(structure: Structure) -> pd.DataFrame:
    """Get mass information for each atom in structure"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    mass_info = pd.DataFrame()
    element_data = get_element_data(structure)
    mass = pd.read_csv(f"{current_dir_path}/datas/Mass.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        mass_value = mass[mass["element"] == element]["atom_mass"].values
        if len(mass_value) != 0:
            mass_info = mass_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "mass": mass_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        m = 0
        if len(index) > 1:
            for i in index:
                m += mass_info[mass_info["index"] == i]["mass"].values.tolist()[0] * mass_info[mass_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            m = mass_info[mass_info["index"] == index[0]]["mass"].values.tolist()[0] * mass_info[mass_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if m != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "mass": m}, ignore_index = True)
    result["element"] = result["element"].apply(split_ion_to_element)
    return result

def get_convalent_radii_info(structure: Structure) -> pd.DataFrame:
    """Get covalent radius information for each atom in structure"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    radii_info = pd.DataFrame()
    element_data = get_element_data(structure)
    radii = pd.read_csv(f"{current_dir_path}/datas/Covalent_Radius.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        radius_value = radii[radii["element"] == element]["covalent_radius"].values
        if len(radius_value) != 0:
            radii_info = radii_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "covalent_radius": radius_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        r = 0
        if len(index) > 1:
            for i in index:
                r += radii_info[radii_info["index"] == i]["covalent_radius"].values.tolist()[0] * radii_info[radii_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            r = radii_info[radii_info["index"] == index[0]]["covalent_radius"].values.tolist()[0] * radii_info[radii_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if r != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "covalent_radius": r}, ignore_index = True)
    return result

def get_convalent_volume_info(structure: Structure) -> pd.DataFrame:
    """Get covalent radius information for each atom in structure"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    radii_info = pd.DataFrame()
    element_data = get_element_data(structure)
    radii = pd.read_csv(f"{current_dir_path}/datas/Covalent_Radius.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        radius_value = radii[radii["element"] == element]["covalent_radius"].values
        if len(radius_value) != 0:
            radii_info = radii_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "covalent_radius": radius_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        v = 0
        if len(index) > 1:
            for i in index:
                v += 4 / 3 * math.pi * ((radii_info[radii_info["index"] == i]["covalent_radius"].values.tolist()[0]) ** 3) * radii_info[radii_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            v = 4 / 3 * math.pi * ((radii_info[radii_info["index"] == index[0]]["covalent_radius"].values.tolist()[0]) ** 3) * radii_info[radii_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if v != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "covalent_volume": v}, ignore_index = True)
    return result

def get_MB_Electronegativity(structure: Structure) -> pd.DataFrame:
    """Getstructureseach inMatyonov-Batsanov electronegativity information for atoms"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    en_info = pd.DataFrame()
    element_data = get_element_data(structure)
    en = pd.read_csv(f"{current_dir_path}/datas/Matyonov-Batsanov_Electronegativity.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        en_value = en[en["elements"] == element]["electronegativity"].values
        if len(en_value) != 0:
            en_info = en_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "electronegativity": en_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        e = 0
        if len(index) > 1:
            for i in index:
                e += en_info[en_info["index"] == i]["electronegativity"].values.tolist()[0] * en_info[en_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            e = en_info[en_info["index"] == index[0]]["electronegativity"].values.tolist()[0] * en_info[en_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if e != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "electronegativity": e}, ignore_index = True)
    return result

def get_valence_Eletronegativity(structure: Structure) -> pd.DataFrame:
    """Get valence electron electronegativity information for each atom"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    en_info = pd.DataFrame()
    element_data = get_element_data(structure)
    Electronegativity = pd.read_csv(f"{current_dir_path}/datas/Electronegativity.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        valence = row["valence"]
        en_value = Electronegativity[(Electronegativity["element"] == element) & (Electronegativity["valence"] == abs(valence))]["en"].values
        if len(en_value) != 0:
            en_info = en_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "valence_electronegativity": en_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        e = 0
        if len(index) > 1:
            for i in index:
                e += en_info[en_info["index"] == i]["valence_electronegativity"].values.tolist()[0] * en_info[en_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            e = en_info[en_info["index"] == index[0]]["valence_electronegativity"].values.tolist()[0] * en_info[en_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if e != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "valence_electronegativity": e}, ignore_index = True)
    return result
    
def get_Pauling_Electronegativity(structure: Structure) -> pd.DataFrame:
    """GetstructuresFor each atom inPaulineElectronegativity information"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    en_info = pd.DataFrame()
    element_data = get_element_data(structure)
    Pauling_en = pd.read_csv(f"{current_dir_path}/datas/Pauling_Electronegativity.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        en_value = Pauling_en[Pauling_en["element"] == element]["Electronegativity"].values
        if len(en_value) != 0:
            en_info = en_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                           "pauling_electronegativity": en_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        e = 0
        if len(index) > 1:
            for i in index:
                e += en_info[en_info["index"] == i]["pauling_electronegativity"].values.tolist()[0] * en_info[en_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            e = en_info[en_info["index"] == index[0]]["pauling_electronegativity"].values.tolist()[0] * en_info[en_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if e != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "pauling_electronegativity": e}, ignore_index = True)
    return result

def get_atomic_radius_info(structure: Structure) -> pd.DataFrame:
    """Get atomic radius information for each atom"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    an_info = pd.DataFrame()
    element_data = get_element_data(structure)
    atomic_radius = pd.read_csv(f"{current_dir_path}/datas/Atomic_Radius.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        an_value = atomic_radius[atomic_radius["element"] == element]["atomic_radius"].values
        if len(an_value) != 0:
            an_info = an_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                        "atomic_radius": an_value[0] / 100}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        a = 0
        if len(index) > 1:
            for i in index:
                a += an_info[an_info["index"] == i]["atomic_radius"].values.tolist()[0] * an_info[an_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            a = an_info[an_info["index"] == index[0]]["atomic_radius"].values.tolist()[0] * an_info[an_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if a != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "atomic_radius": a}, ignore_index = True)
    return result

def get_atomic_volume_info(structure: Structure) -> pd.DataFrame:
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    an_info = pd.DataFrame()
    element_data = get_element_data(structure)
    atomic_radius = pd.read_csv(f"{current_dir_path}/datas/Atomic_Radius.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        an_value = atomic_radius[atomic_radius["element"] == element]["atomic_radius"].values
        if len(an_value) != 0:
            an_info = an_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                        "atomic_radius": an_value[0] / 100}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        a = 0
        if len(index) > 1:
            for i in index:
                a += 4 / 3 * math.pi * ((an_info[an_info["index"] == i]["atomic_radius"].values.tolist()[0]) ** 3) * an_info[an_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            a = 4 / 3 * math.pi * ((an_info[an_info["index"] == index[0]]["atomic_radius"].values.tolist()[0]) ** 3) * an_info[an_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if a != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "atomic_volume": a}, ignore_index = True)
    return result

def get_atomic_number_info(structure: Structure) -> pd.DataFrame:
    """Get atomic number information for each atom"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    an_info = pd.DataFrame()
    element_data = get_element_data(structure)
    atomic_number = pd.read_csv(f"{current_dir_path}/datas/Atomic_Number.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        an_value = atomic_number[atomic_number["element"] == element]["atomic_number"].values
        if len(an_value) != 0:
            an_info = an_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                        "atomic_number": an_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        a = 0
        if len(index) > 1:
            for i in index:
                a += an_info[an_info["index"] == i]["atomic_number"].values.tolist()[0] * an_info[an_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            a = an_info[an_info["index"] == index[0]]["atomic_number"].values.tolist()[0] * an_info[an_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if a != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "atomic_number": a}, ignore_index = True)
    return result

def get_ionisation_energy_info(structure: Structure) -> pd.DataFrame:
    """Get ionization energy information for each atom"""
    from features.env import get_dope_index_info
    index_info = get_dope_index_info(structure)
    ie_info = pd.DataFrame()
    element_data = get_element_data(structure)
    ionisation_energy = pd.read_csv(f"{current_dir_path}/datas/Ionisation_Energy.csv")
    for i, row in element_data.iterrows():
        element = row["element"]
        element = split_ion_to_element(str(element))
        ie_value = ionisation_energy[ionisation_energy["element"] == element]["ionisation_energy"].values
        if len(ie_value) != 0:
            ie_info = ie_info._append({"index": row["index"], "element": element, "ratio": row["ratio"], 
                                        "ionisation_energy": ie_value[0]}, ignore_index = True)
    result = pd.DataFrame()
    for ind in index_info:
        index = index_info[ind]
        ie = 0
        if len(index) > 1:
            for i in index:
                ie += ie_info[ie_info["index"] == i]["ionisation_energy"].values.tolist()[0] * ie_info[ie_info["index"] == i]["ratio"].values.tolist()[0]
        else:
            ie = ie_info[ie_info["index"] == index[0]]["ionisation_energy"].values.tolist()[0] * ie_info[ie_info["index"] == index[0]]["ratio"].values.tolist()[0]
        if ie != 0:
            result = result._append({"index": ind, "element": structure[ind].species_string, "ionisation_energy": ie}, ignore_index = True)
    return result


import re

def split_ion_to_element(input_str):
    # if ":" in input_str:
        # input_str = input_str.split(":")[0]
    # Regex 1: match pure element symbols (no charge), e.g., Ba, Ca, Al, O, Fe
    pure_element_pattern = r'^[A-Z][a-z]?$'
    # Regex 2: match charged ion strings, e.g., Ba2+, Ca2+, Al3+, O2-
    ion_pattern = r'^([A-Z][a-z]?)(\d*[+-])$'
    # First check if it's a pure element symbol
    if re.match(pure_element_pattern, input_str):
        return input_str  # Return pure element symbol as-is
    # Then check if it's a charged ion string
    ion_match = re.match(ion_pattern, input_str)
    if ion_match:
        return ion_match.group(1)  # Extract element symbol
    # Non-element-symbol format (e.g., 123, abc, Ba-) returns None
    return input_str