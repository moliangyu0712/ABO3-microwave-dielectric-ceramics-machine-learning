from pymatgen.core import Structure
from features.valence import get_bond_valences
import pandas as pd
import os
from pymatgen.core.sites import Site
from features.composition_fixed import fixed_species_string
Site.species_string = fixed_species_string

current_dir_path = os.path.dirname(os.path.abspath(__file__))

def get_coordinate_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_cationic_coordination_numbers, get_anionic_coordination_numbers
    from features.env import get_dope_index_info
    from features.valence import get_valence
    cationic_coord_info = get_cationic_coordination_numbers(structure, formula, spn)
    anionic_coord_info = get_anionic_coordination_numbers(structure, formula, spn)
    index_info = get_dope_index_info(structure)
    valences_info  = get_valence(structure)
    coordinate_info = {}
    for key in index_info:
        index = index_info[key]
        coord = 0
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences_info[key][ind] > 0:
                    coord += cationic_coord_info[cationic_coord_info["src_index"] == i]["coordination_number"].values.tolist()[0]
                else:
                    coord += anionic_coord_info[anionic_coord_info["dst_index"] == i]["coordination_number"].values.tolist()[0]
            coord = coord / len(index)
        else:
            if valences_info[key][0] > 0:
                coord = cationic_coord_info[cationic_coord_info["src_index"] == index[0]]["coordination_number"].values.tolist()[0]
            else:
                coord += anionic_coord_info[anionic_coord_info["dst_index"] == index[0]]["coordination_number"].values.tolist()[0]
        if coord != 0:
            coordinate_info[key] = coord
    return coordinate_info

def get_bond_length_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_bond_info
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    bond_length_info = pd.DataFrame()
    for n, row in bond_info.iterrows():
        valence = row['valence_src']
        if valence > 0:
            bond_length_info = bond_length_info._append({"src_index":row['src_index'],
                                                        "src":row['src'],
                                                        "src_ratio":row['src_ratio'],
                                                        "dst":row['dst'],
                                                        "distance":row['distance']}, ignore_index=True)
    return bond_length_info

def get_average_bond_length(structure: Structure, formula: None, spn = None) -> dict:
    """Get average bond length for each cation in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    bond_length_info = get_bond_length_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    bond_length_info["mean_distance"] = bond_length_info.groupby('src_index')['distance'].transform('mean')
    bond_length_info1 = bond_length_info.drop_duplicates(subset=['src_index'])
    average_bond_length = {}
    for key in index_info:
        index = index_info[key]
        mbl = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bl = bond_length_info1[bond_length_info1['src_index'] == i]['mean_distance'].values.tolist()
                    r = bond_length_info1[bond_length_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbl += bl[0] * r[0]
        else:
            if valences[key][0] > 0:
                bl = bond_length_info1[bond_length_info1['src_index'] == index[0]]['mean_distance'].values.tolist()
                r = bond_length_info1[bond_length_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbl = bl[0] * r[0]
        if mbl != 0:
            average_bond_length[key] = mbl
    return average_bond_length

def get_stand_dev_bond_length(structure: Structure, formula = None, spn = None) -> dict:
    """Get bond length standard deviation for each cation in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    bond_length_info = get_bond_length_info(structure, formula = formula, spn= spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    bond_length_info["std_distance"] = bond_length_info.groupby('src_index')['distance'].transform('std')
    bond_length_info1 = bond_length_info.drop_duplicates(subset=['src_index'])
    std_bond_length = {}
    for key in index_info:
        index = index_info[key]
        mbl = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bl = bond_length_info1[bond_length_info1['src_index'] == i]['std_distance'].values.tolist()
                    r = bond_length_info1[bond_length_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbl += bl[0] * r[0]
        else:
            if valences[key][0] > 0:
                bl = bond_length_info1[bond_length_info1['src_index'] == index[0]]['std_distance'].values.tolist()
                r = bond_length_info1[bond_length_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbl = bl[0] * r[0]
        if mbl != 0:
            std_bond_length[key] = mbl
    return std_bond_length

def bond_strength_covalence_caculate(element, valence, distance):
    aM = pd.read_csv(f'{current_dir_path}/datas/Covalence.csv')
    blls = pd.read_csv(f'{current_dir_path}/datas/Bond_Length.csv')
    s = 0
    # Bond strength calculation
    R1 = blls[(blls["cation"] == element) & (blls["valence"] == valence)]["R1(A)"].values[0]
    N = blls[(blls["cation"] == element) & (blls["valence"] == valence)]["N"].values[0]
    s = ( distance / R1 )**( - N )
    f = 0
    # Covalence calculation
    a = aM[aM["element"] == element]["a"].values[0]
    M = aM[aM["element"] == element]["M"].values[0]
    f = a * ( s ** (M) )
    # Covalence degree calculation
    cvd = f / s
    result_list = [s, f, cvd]
    return result_list

def get_bond_strength_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_bond_info
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    bond_strength_info = pd.DataFrame()
    for n, row in bond_info.iterrows():
        src = row['src']
        valence = row['valence_src']
        dist = row['distance']
        bs = bond_strength_covalence_caculate(src, valence, dist)
        bond_strength_info = bond_strength_info._append({"src_index":row['src_index'],
                                                        "src":src,
                                                        "src_ratio":row['src_ratio'],
                                                        "dst":row['dst'],
                                                        "distance":dist,
                                                        "bond_strength":bs[0]}, ignore_index=True)
    return bond_strength_info

def get_total_bond_strength(structure: Structure, formula = None, spn = None) -> dict:
    """Get total bond strength for each atom in structure"""
    from features.env import get_dope_index_info
    bond_strength_info = get_bond_strength_info(structure, formula = formula, spn = spn) 
    ele_bs = {}
    for n, row in bond_strength_info.iterrows():
        if row["src_index"] not in ele_bs:
            ele_bs[row["src_index"]] = row["bond_strength"] * row["src_ratio"]
        else:
            ele_bs[row["src_index"]] += row["bond_strength"] * row["src_ratio"]
    index_info = get_dope_index_info(structure)
    result = {}
    for key in index_info:
        total_bv = 0
        if len(index_info[key]) > 1:
            for idx in index_info[key]:
                total_bv += ele_bs.get(idx, 0)
        else:
            total_bv = ele_bs.get(index_info[key][0], 0)
        if total_bv != 0:
            result[key] = total_bv
    
    return result

def get_average_bond_strength(structure: Structure, formula = None, spn = None) -> dict:
    from features.env import get_dope_index_info
    from features.valence import get_valence
    bond_strength_info = get_bond_strength_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    bond_strength_info["mean_bs"] = bond_strength_info.groupby('src_index')['bond_strength'].transform('mean')
    bond_strength_info1 = bond_strength_info.drop_duplicates(subset=['src_index'])
    average_bond_strength = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = bond_strength_info1[bond_strength_info1['src_index'] == i]['mean_bs'].values.tolist()
                    r = bond_strength_info1[bond_strength_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = bond_strength_info1[bond_strength_info1['src_index'] == index[0]]['mean_bs'].values.tolist()
                r = bond_strength_info1[bond_strength_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            average_bond_strength[key] = mbs
    return average_bond_strength

def get_stand_dev_bond_strength(structure: Structure, formula = None, spn = None) -> dict:
    """Get bond strength standard deviation for each atom in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    bond_strength_info = get_bond_strength_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    bond_strength_info["std_bs"] = bond_strength_info.groupby('src_index')['bond_strength'].transform('std')
    bond_strength_info1 = bond_strength_info.drop_duplicates(subset=['src_index'])
    std_bond_strength = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = bond_strength_info1[bond_strength_info1['src_index'] == i]['std_bs'].values.tolist()
                    r = bond_strength_info1[bond_strength_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = bond_strength_info1[bond_strength_info1['src_index'] == index[0]]['std_bs'].values.tolist()
                r = bond_strength_info1[bond_strength_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            std_bond_strength[key] = mbs
    return std_bond_strength

def get_covalence_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_bond_info
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    covalence_info = pd.DataFrame()
    for n, row in bond_info.iterrows():
        src = row['src']
        valence = row['valence_src']
        dist = row['distance']
        bs = bond_strength_covalence_caculate(src, valence, dist)
        covalence_info = covalence_info._append({"src_index":row['src_index'],
                                                "src":src,
                                                "src_ratio":row['src_ratio'],
                                                "dst":row['dst'],
                                                "distance":dist,
                                                "covalence":bs[1]}, ignore_index=True)
    return covalence_info

def get_average_covalence(structure: Structure, formula = None, spn = None) -> dict:
    """Get total covalence for each atom in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    covalence_info = get_covalence_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    covalence_info["mean_covalence"] = covalence_info.groupby('src_index')['covalence'].transform('mean')
    covalence_info1 = covalence_info.drop_duplicates(subset=['src_index'])
    average_covalence = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = covalence_info1[covalence_info1['src_index'] == i]['mean_covalence'].values.tolist()
                    r = covalence_info1[covalence_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = covalence_info1[covalence_info1['src_index'] == index[0]]['mean_covalence'].values.tolist()
                r = covalence_info1[covalence_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            average_covalence[key] = mbs
    return average_covalence

def get_stand_dev_covalence(structure: Structure, formula = None, spn = None) -> dict:
    """Get covalence standard deviation for each atom in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    covalence_info = get_covalence_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    covalence_info["std_covalence"] = covalence_info.groupby('src_index')['covalence'].transform('std')
    covalence_info1 = covalence_info.drop_duplicates(subset=['src_index'])
    std_covalence = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = covalence_info1[covalence_info1['src_index'] == i]['std_covalence'].values.tolist()
                    r = covalence_info1[covalence_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = covalence_info1[covalence_info1['src_index'] == index[0]]['std_covalence'].values.tolist()
                r = covalence_info1[covalence_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            std_covalence[key] = mbs
    return std_covalence

def get_cvd_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.env import get_bond_info
    bond_info = get_bond_info(structure, dope = True, formula = formula, spn = spn)
    cvd_info = pd.DataFrame()
    for n, row in bond_info.iterrows():
        src = row['src']
        valence = row['valence_src']
        dist = row['distance']
        bs = bond_strength_covalence_caculate(src, valence, dist)
        cvd_info = cvd_info._append({"src_index":row['src_index'],
                                    "src":src,
                                    "src_ratio":row['src_ratio'],
                                    "dst":row['dst'],
                                    "distance":dist,
                                    "cvd":bs[2]}, ignore_index=True)
    return cvd_info

def get_average_cvd(structure: Structure, formula = None, spn = None) -> dict:
    """Get total covalence degree for each atom in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    cvd_info = get_cvd_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    cvd_info["mean_cvd"] = cvd_info.groupby('src_index')['cvd'].transform('mean')
    cvd_info1 = cvd_info.drop_duplicates(subset=['src_index'])
    average_cvd = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = cvd_info1[cvd_info1['src_index'] == i]['mean_cvd'].values.tolist()
                    r = cvd_info1[cvd_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = cvd_info1[cvd_info1['src_index'] == index[0]]['mean_cvd'].values.tolist()
                r = cvd_info1[cvd_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            average_cvd[key] = mbs
    return average_cvd

def get_stand_dev_cvd(structure: Structure, formula = None, spn = None) -> dict:
    """Get covalence degree standard deviation for each atom in structure"""
    from features.env import get_dope_index_info
    from features.valence import get_valence
    cvd_info = get_cvd_info(structure, formula = formula, spn = spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    cvd_info["std_cvd"] = cvd_info.groupby('src_index')['cvd'].transform('std')
    cvd_info1 = cvd_info.drop_duplicates(subset=['src_index'])
    std_cvd = {}
    for key in index_info:
        index = index_info[key]
        mbs = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    bs = cvd_info1[cvd_info1['src_index'] == i]['std_cvd'].values.tolist()
                    r = cvd_info1[cvd_info1['src_index'] == i]['src_ratio'].values.tolist()
                    mbs += bs[0] * r[0]
        else:
            if valences[key][0] > 0:
                bs = cvd_info1[cvd_info1['src_index'] == index[0]]['std_cvd'].values.tolist()
                r = cvd_info1[cvd_info1['src_index'] == index[0]]['src_ratio'].values.tolist()
                mbs = bs[0] * r[0]
        if mbs != 0:
            std_cvd[key] = mbs
    return std_cvd

def get_fi_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.lattice_energy import LatticeEnergyCalculator
    from features.env import get_dope_index_info
    from features.valence import get_valence
    LEC = LatticeEnergyCalculator(structure, formula, spn)
    fi_info = LEC.calc_lattice_energy()[1]
    fi_info = pd.DataFrame(fi_info)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    fi_info["mean_fi"] = fi_info.groupby('src_index')['fi'].transform('mean')
    fi_info1 = fi_info.drop_duplicates(subset=['src_index'])
    average_fi = {}
    for key in index_info:
        index = index_info[key]
        mfi = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    fi = fi_info1[fi_info1['src_index'] == i]['mean_fi'].values.tolist()
                    r = fi_info1[fi_info1['src_index'] == i]['ratio'].values.tolist()
                    mfi += fi[0] * r[0]
        else:
            if valences[key][0] > 0:
                fi = fi_info1[fi_info1['src_index'] == index[0]]['mean_fi'].values.tolist()
                r = fi_info1[fi_info1['src_index'] == index[0]]['ratio'].values.tolist()
                mfi = fi[0] * r[0]
        if mfi != 0:
            average_fi[key] = mfi
    return average_fi

    return fi_info
def get_U_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.lattice_energy import LatticeEnergyCalculator
    from features.env import get_dope_index_info
    from features.valence import get_valence
    LEC = LatticeEnergyCalculator(structure, formula, spn)
    U_info = LEC.calc_lattice_energy()[1]
    U_info = pd.DataFrame(U_info)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    U_info["mean_U"] = U_info.groupby('src_index')['U_bond'].transform('mean')
    U_info1 = U_info.drop_duplicates(subset=['src_index'])
    average_U = {}
    for key in index_info:
        index = index_info[key]
        mU = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    U = U_info1[U_info1['src_index'] == i]['mean_U'].values.tolist()
                    r = U_info1[U_info1['src_index'] == i]['ratio'].values.tolist()
                    mU += U[0] * r[0]
        else:
            if valences[key][0] > 0:
                U = U_info1[U_info1['src_index'] == index[0]]['mean_U'].values.tolist()
                r = U_info1[U_info1['src_index'] == index[0]]['ratio'].values.tolist()
                mU = U[0] * r[0]
        if mU != 0:
            average_U[key] = mU
    return average_U

def get_bond_energy_info(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.bond_energy import get_bond_energy
    from features.env import get_dope_index_info
    from features.valence import get_valence
    bond_energy_info = get_bond_energy(structure, formula, spn)
    index_info = get_dope_index_info(structure)
    valences = get_valence(structure)
    bond_energy_info["mean_bond_energy"] = bond_energy_info.groupby('src_index')['Energy'].transform('mean')
    bond_energy_info1 = bond_energy_info.drop_duplicates(subset=['src_index'])
    average_bond_energy = {}
    for key in index_info:
        index = index_info[key]
        mbe = 0 
        if len(index) > 1:
            for ind, i in enumerate(index):
                if valences[key][ind] > 0:
                    be = bond_energy_info1[bond_energy_info1['src_index'] == i]['mean_bond_energy'].values.tolist()
                    r = bond_energy_info1[bond_energy_info1['src_index'] == i]['ratio'].values.tolist()
                    mbe += be[0] * r[0]
        else:
            if valences[key][0] > 0:
                be = bond_energy_info1[bond_energy_info1['src_index'] == index[0]]['mean_bond_energy'].values.tolist()
                r = bond_energy_info1[bond_energy_info1['src_index'] == index[0]]['ratio'].values.tolist()
                mbe = be[0] * r[0]
        if mbe != 0:
            average_bond_energy[key] = mbe
    return average_bond_energy


def get_bonds_information(structure: Structure, formula = None, spn = None) -> pd.DataFrame:
    from features.element_info import split_ion_to_element
    bs_mean = get_average_bond_strength(structure, formula, spn)
    bl_mean = get_average_bond_length(structure, formula, spn)
    cov_mean = get_average_covalence(structure, formula, spn)
    cvd_mean = get_average_cvd(structure, formula, spn)
    bs_std = get_stand_dev_bond_strength(structure, formula, spn)
    bl_std = get_stand_dev_bond_length(structure, formula, spn)
    cov_std = get_stand_dev_covalence(structure, formula, spn)
    cvd_std = get_stand_dev_cvd(structure, formula, spn)
    cn = get_coordinate_info(structure, formula, spn)
    bvs = get_bond_valences(structure, formula, spn)[0]
    fi = get_fi_info(structure, formula, spn)
    U_mean = get_U_info(structure, formula, spn)
    E_mean = get_bond_energy_info(structure, formula, spn)
    elements = {}
    for i, site in enumerate(structure):
        elements[i] = site.species_string
    result = pd.DataFrame()
    for key in elements:
        result = result._append({"index":key,
                                "element":elements[key],
                                "coordination_number":cn.get(key, 0),
                                "avg_bond_length":bl_mean.get(key, 0),
                                "std_bond_length":bl_std.get(key, 0),
                                "bvs":bvs.get(key, 0),
                                "avg_bond_strength":bs_mean.get(key, 0),
                                "std_bond_strength":bs_std.get(key, 0),
                                "avg_covalence":cov_mean.get(key, 0),
                                "std_covalence":cov_std.get(key, 0),
                                "avg_cvd":cvd_mean.get(key, 0),
                                "std_cvd":cvd_std.get(key, 0),
                                "avg_fi":fi.get(key, 0),
                                "avg_U":U_mean.get(key, 0),
                                "avg_E":E_mean.get(key, 0)}, 
                                ignore_index=True)
    
    numeric_columns = [col for col in result.columns if col not in ["index", "element"]]
    result = result[result[numeric_columns].ne(0).any(axis=1)]
    result = result.reset_index(drop=True)
    result["element"] = result["element"].apply(split_ion_to_element)
    return result