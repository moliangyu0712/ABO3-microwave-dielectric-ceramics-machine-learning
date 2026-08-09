import numpy as np
import pandas as pd
from pymatgen.core import Structure
import os
current_dir_path = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings("ignore")
# Display all rows
pd.set_option('display.max_rows', None)
# Display all columns
pd.set_option('display.max_columns', None)
import re
def split_string_by_uppercase(string):
    pattern = r'([A-Z][a-z0-9.]*)'  # Match uppercase letter followed by lowercase letters or digits
    split_strings = re.findall(pattern, string)
    return split_strings
def separate_letters_and_numbers(string):
    pattern = r'[a-zA-Z]+|\d+(?:\.\d+)?'  # Match letters or digits (including decimals)
    separate_parts = re.findall(pattern, string)
    return separate_parts

class LatticeEnergyCalculator:
    """
    Ionic crystal lattice energy calculator based on chemical bond theory
    Revised version: full Angstrom (Å) unit system, avoiding extremely small numerical operations
    """
    def __init__(self, structure: Structure, formula: str, spn: int):
        self.struct = structure
        # ✅ 1. Unit cell volume directly uses Å³
        self.V_cell_ang3 = self.struct.volume
        
        # Paper parameters (unchanged)
        self.B = 2100
        self.C_exp = 1.64
        self.D = 0.75
        self.k1 = 1270
        self.beta = 0.089
        self.aB = 0.5292  # Bohr radius (Å)
        self.formula = formula
        self.spn = spn
        formu = (self.struct.formula).replace(" ","")
        O_n = float(separate_letters_and_numbers(split_string_by_uppercase(formu)[-1])[-1])
        O_p = float(separate_letters_and_numbers(split_string_by_uppercase(self.formula)[-1])[-1])
        self.N = O_n/O_p

    def get_bond_info(self):
        try:
            from features.env import get_bond_info as get_bonds
            bond_info = get_bonds(self.struct, dope=True, formula=self.formula, spn=self.spn)
            bond_info.rename(columns={
            'src': 'atom1', 'dst': 'atom2', 'distance': 'd', 
            'valence_src': 'z1', 'valence_dst': 'z2', 
            'src_index':'n1', 'dst_index':'n2', 'src_ratio': 'r1', 'dst_ratio': 'r2'
            }, inplace=True)
            bond_info['cn1'] = bond_info.groupby('n1')['n1'].transform('count')
            bond_info['cn2'] = bond_info.groupby('n2')['n2'].transform('count')
            bond_info['d'] = bond_info['d'].round(6)
            bond_info['N_2-1'] = bond_info.groupby(['dst_origin_index', 'd', 'n1'])['d'].transform('count')
            bond_info['N_1-2'] = bond_info.groupby(['src_origin_index', 'd','n2'])['d'].transform('count')
            bond_info['src_N'] = bond_info.groupby(['atom1', 'src_origin_index'])['src_origin_index'].transform('count') / bond_info["cn1"] / self.N
            bond_info['dst_N'] = bond_info.groupby(['atom2', 'dst_origin_index'])['dst_origin_index'].transform('count') / bond_info["cn2"] / self.N
            bond_info["m"] = bond_info['N_2-1'] / bond_info["cn1"] * bond_info['src_N']
            bond_info["n"] = bond_info['N_1-2'] / bond_info["cn2"] * bond_info['dst_N']
            bond_info["Z_p"] = bond_info["z1"]
            bond_info["Z_n"] = bond_info["Z_p"] * bond_info["m"] / bond_info["n"]
            return bond_info
        except ImportError:
            # If module not available, this is demo only; replace with actual data retrieval in practice
            print("Warning: features.env not found. Please ensure your environment is set up correctly.")
            return pd.DataFrame()

    def _get_valence_electrons(self, elem):
        # Supplement missing methods
        valence_e = {
            # Main group
            "H": 1,
            "Li": 1, "Be": 2,
            "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
            "Na": 1, "Mg": 2,
            "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
            "K": 1, "Ca": 2,
            "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
            "Rb": 1, "Sr": 2,
            "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
            "Cs": 1, "Ba": 2,
            "Tl": 3, "Pb": 4, "Bi": 5, "Po": 6, "At": 7,
            "Fr": 1, "Ra": 2,

            # Subgroup · Period 4
            "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
            "Fe": 8, "Co": 9, "Ni": 10,
            "Cu": 11, "Zn": 12,

            # Subgroup · Period 5
            "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7,
            "Ru": 8, "Rh": 9, "Pd": 10,
            "Ag": 11, "Cd": 12,

            # Subgroup · Period 6    
            "La": 3, "Hf": 4, "Ta": 5, "W": 6, "Re": 7,
            "Os": 8, "Ir": 9, "Pt": 10,
            "Au": 11, "Hg": 12,

            # Subgroup · Period 7
            "Ac": 3, "Rf": 4, "Db": 5, "Sg": 6, "Bh": 7,
            "Hs": 8, "Mt": 9, "Ds": 10,
            "Rg": 11, "Cn": 12,

            # Lanthanides
            "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3,
            "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3,
            "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,

            # Actinides (may be inaccurate, use with caution!)
            "Th": 4, "Pa": 5, "U": 6, "Np": 5, "Pu": 6,
            "Am": 3, "Cm": 3, "Bk": 3, "Cf": 3, "Es": 3,
            "Fm": 3, "Md": 3, "No": 3, "Lr": 3
        }
        return valence_e.get(elem, 0)

    def _compute_Nb_and_vb(self, all_bonds_df):
        """
        Calculate Nb (bond density) and vb (bond volume)
        Unit system: full Angstrom (Å)
        - V_cell: Å³
        - Nb: /Å³
        - vb: Å³
        """
        # 1. Get unit cell volume (Å³)
        V_cell = self.V_cell_ang3

        # 2. Cluster by (n1, n2)
        grouped = all_bonds_df.groupby(['n1', 'n2'])
        bond_stats = {}
        
        for (n1, n2), group in grouped:
            key = (n1, n2)
            count_in_cell = len(group)
            
            # ✅ 2. Calculate Nb: Unit is /Å³
            Nb_ang3 = count_in_cell / V_cell
            
            d_mean = group['d'].mean()
            bond_stats[key] = {'Nb': Nb_ang3, 'd': d_mean}

        # 3. CalculateDenominator: Σ (d³ * Nb)
        # Unit analysis: (Å³) * (/Å³) =  
        sum_d3_Nb = 0.0
        for key, stats in bond_stats.items():
            sum_d3_Nb += (stats['d'] ** 3) * stats['Nb']

        # 4. Calculate vb
        # Formula: vb = d³ / Σ(d³ * Nb)
        # Unit analysis: Å³ / (Dimensionless) = Å³
        vb_list = []
        
        for idx, row in all_bonds_df.iterrows():
            n1, n2 = row['n1'], row['n2']
            d = row['d']
            key = (n1, n2)
            stats = bond_stats[key]
            # ✅ 3. Calculate vb (Unit is Å³)
            vb_val = (d ** 3) / sum_d3_Nb
            vb_list.append(vb_val)
        all_bonds_df['vb'] = vb_list
        return all_bonds_df

    def _compute_ks(self, vb, ZA_star, ZB_star, N_CA, N_CB):
        ne_star = (ZA_star / N_CA) + (ZB_star / N_CB)
        Ne_star = ne_star / vb  
        kF = (3 * np.pi**2 * Ne_star) ** (1/3)
        ks = np.sqrt(4 * kF / (np.pi * self.aB))
        return ks

    def calc_ionicity(self, d, vb, N_CA, N_CB, m, n, z_c, z_a, atom1, atom2):
        # Here d Is Å
        Eh = 39.74 / (d ** 2.48)
        r0 = d / 2.0
        Nc = (m/(m+n))*N_CA + (n/(m+n))*N_CB
        b = self.beta * (Nc ** 2)
        ZA_star = self._get_valence_electrons(atom1) * z_c / self._get_valence_electrons(atom1)
        ZB_star = self._get_valence_electrons(atom2) * z_a / (self.get_N_in_ZB_star(atom2) - self._get_valence_electrons(atom2))
        
        # Calculate ks (Unit 1/Å)
        ks = self._compute_ks(vb, ZA_star, ZB_star, N_CA, N_CB)
        
        # r0 Is Å,ks Is 1/Å,Product is dimensionless,exp CalculateSafety
        exp_term = np.exp(-ks * r0)

        if n > m:
            inside = (ZA_star / r0) - (n/m)*(ZB_star / r0)
        elif m > n:
            inside = (m/n)*(ZA_star / r0) - (ZB_star / r0)
        else:
            inside = (ZA_star - ZB_star) / r0

        C = 14.4 * b * exp_term * inside
        Eg = np.sqrt(Eh**2 + C**2)

        if Eg == 0:
            return 0.95, 0.05
        fi = abs(C**2 / Eg**2)
        fc = abs(Eh**2 / Eg**2)
        return fi, fc

    def get_N_in_ZB_star(self, element):
        if element == "H":
            N = 2
        elif element == "He":
            N = 2
        else:
            N = 8
        return N
    
    def calc_lattice_energy(self):
        bonds = self.get_bond_info()
        if bonds.empty:
            return 0, []
        
        bonds = self._compute_Nb_and_vb(bonds)
        
        total_U = 0
        bond_results = []
        
        for _, b in bonds.iterrows():
            d = b['d']
            index1 = b['n1']
            index2 = b['n2']
            r1 = b['r1']
            r2 = b['r2']
            cn1 = b['cn1']
            cn2 = b['cn2']
            vb = b['vb']
            atom1 = b['atom1']
            atom2 = b['atom2']
            m = b['m']
            n = b['n']
            z_c = b['Z_p']
            z_a = b['Z_n']
            self.current_cation = atom1
            self.current_anion = atom2
            N_CA, N_CB = cn1, cn2
            
            fi, fc = self.calc_ionicity(d, vb, N_CA, N_CB, m, n, z_c, z_a, atom1, atom2)
            Ui = self.k1 * (m + n) * z_c * z_a / d * (1 - 0.4/d) * fi
            Uc = self.B * m * (z_c ** self.C_exp) / (d ** self.D) * fc
            U_bond = Ui + Uc
            
            bond_results.append({
                'src_index': index1,
                'src': atom1,
                'dst': atom2,
                'dst_index': index2,
                'bond': f"{atom1}-{atom2}",
                'ratio': round(r1, 5),
                'd(Å)': round(d, 5),
                'fi': round(fi, 5),
                'Ui(kJ/mol)': round(Ui, 1),
                'Uc(kJ/mol)': round(Uc, 1),
                'U_bond': round(U_bond, 1)
            })
            total_U += U_bond * r1
        total_U = total_U / self.N
        return round(total_U, 1), bond_results
    
    def get_U(self):
        lattice_energy, bond_details = self.calc_lattice_energy()
        bond_details = pd.DataFrame(bond_details)
        result = bond_details.drop_duplicates(subset=['src', 'd(Å)'], keep='first')
        U = 0
        for idx, row in result.iterrows():
            U += row['U_bond']*row['ratio']
        return U, lattice_energy, bond_details

    def get_lattice_energy_info(self):
        U, lattice_energy, bond_details = self.get_U()
        return U, lattice_energy, bond_details
        