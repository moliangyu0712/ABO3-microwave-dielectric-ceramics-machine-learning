from features.env import get_bond_info as get_bonds
import os
import pandas as pd
from pymatgen.core import Structure
import math
current_dir_path = os.path.dirname(os.path.abspath(__file__))
cov_rad = pd.read_csv(f"{current_dir_path}/datas/Covalent_Radius.csv")
ele_Pauling = pd.read_csv(f"{current_dir_path}/datas/Pauling_Electronegativity.csv")
Exx = pd.read_csv(f"{current_dir_path}/datas/isonuclear_bond_energy.csv")

import re
def split_string_by_uppercase(string):
    pattern = r'([A-Z][a-z0-9.]*)'  # Match uppercase letter followed by lowercase letters or digits
    split_strings = re.findall(pattern, string)
    return split_strings
def separate_letters_and_numbers(string):
    pattern = r'[a-zA-Z]+|\d+(?:\.\d+)?'  # Match letters or digits (including decimals)
    separate_parts = re.findall(pattern, string)
    return separate_parts

def calculate_bond_energy(SA, SB, delta_S, d, cov_rad_A, cov_rad_B, EAA, EBB):
    ti = abs( (SA - SB) / delta_S / 2 )
    tc = 1 - ti
    Ei = 1389.088 / d
    Ec = (cov_rad_A + cov_rad_B) / d * math.sqrt(EAA * EBB)
    total_energy = tc * Ec + ti * Ei
    return total_energy
    
def get_bond_energy(structure: Structure, formula = None, spn = None):
    bond_info = get_bonds(structure, dope=True, formula=formula, spn=spn)
    result = pd.DataFrame()
    for index, row in bond_info.iterrows():
        SA = ele_Pauling[ele_Pauling["element"] == row["src"]]["Electronegativity"].values[0]
        SB = ele_Pauling[ele_Pauling["element"] == row["dst"]]["Electronegativity"].values[0]
        d= row["distance"]
        delta_S = 3
        cov_rad_A = cov_rad[cov_rad["element"] == row["src"]]["covalent_radius"].values[0]
        cov_rad_B = cov_rad[cov_rad["element"] == row["dst"]]["covalent_radius"].values[0]
        EAA = Exx[Exx["element"] == row["src"]]["BDEs"].values[0]
        EBB = Exx[Exx["element"] == row["dst"]]["BDEs"].values[0]
        ratio = row["src_ratio"]
        src_index = row["src_index"]
        dst_index = row["dst_index"]
        Energy = calculate_bond_energy(SA, SB, delta_S, d, cov_rad_A, cov_rad_B, EAA=EAA, EBB=EBB)
        result = result._append({"src_index": src_index, 
                                 "src": row["src"],
                                 "dst_index": dst_index, 
                                 "dst": row["dst"],
                                 "distance": d,
                                 "Energy": Energy, 
                                 "ratio": ratio}, ignore_index=True)
    return result

def get_total_bond_energy(structure: Structure, formula = None, spn = None):
    bond_energy_info = get_bond_energy(structure, formula, spn)
    result = bond_energy_info.drop_duplicates(subset=['src', 'distance'], keep='first')
    E = 0
    for index, row in result.iterrows():
        E += row["Energy"] * row["ratio"] 
    formu = (structure.formula).replace(" ","")
    O_n = float(separate_letters_and_numbers(split_string_by_uppercase(formu)[-1])[-1])
    O_p = float(separate_letters_and_numbers(split_string_by_uppercase(formula)[-1])[-1])
    N = O_n/O_p
    EE = (bond_energy_info["Energy"] * bond_energy_info["ratio"]).sum() / N
    return EE, E