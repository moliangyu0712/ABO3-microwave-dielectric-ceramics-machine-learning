import pandas as pd
from pymatgen.core import Structure
from features.element_info import get_mass_info
from features.element_info import get_convalent_radii_info
from features.element_info import get_MB_Electronegativity
from features.element_info import get_valence_Eletronegativity
from features.element_info import get_Pauling_Electronegativity
from features.element_info import get_atomic_radius_info
from features.element_info import get_atomic_number_info
from features.element_info import get_ionisation_energy_info
from features.element_info import get_convalent_volume_info
from features.element_info import get_atomic_volume_info
from features.dielectric_radius import get_averaged_dielectric_info
from features.bond_info import get_bonds_information
def get_element_info(structure: Structure, formula = None, spn = None):
    mass_info = get_mass_info(structure)
    convalent_radii_info = get_convalent_radii_info(structure)
    convalent_volume_info = get_convalent_volume_info(structure)
    atomic_volume_info = get_atomic_volume_info(structure)
    MB_Electronegativity_info = get_MB_Electronegativity(structure)
    valence_Eletronegativity_info = get_valence_Eletronegativity(structure)
    Pauling_Electronegativity_info = get_Pauling_Electronegativity(structure)
    atomic_radius_info = get_atomic_radius_info(structure)
    atomic_number_info = get_atomic_number_info(structure)
    ionisation_energy_info = get_ionisation_energy_info(structure)
    bond_info = get_bonds_information(structure, formula, spn)
    IR_DP_VI = get_averaged_dielectric_info(structure, formula, spn)
    element_info = pd.DataFrame()
    element_info["index"] = mass_info["index"]
    element_info["element"] = mass_info["element"]
    element_info["mass"] = mass_info["mass"]
    element_info["covalent_radius"] = convalent_radii_info["covalent_radius"]
    element_info["convalent_volume"] = convalent_volume_info["covalent_volume"]
    element_info["MB_Electronegativity"] = MB_Electronegativity_info["electronegativity"]
    element_info["valence_Eletronegativity"] = valence_Eletronegativity_info["valence_electronegativity"]
    element_info["Pauling_Electronegativity"] = Pauling_Electronegativity_info["pauling_electronegativity"]
    element_info["atomic_radius"] = atomic_radius_info["atomic_radius"]
    element_info["atomic_volume"] = atomic_volume_info["atomic_volume"]
    element_info["atomic_number"] = atomic_number_info["atomic_number"]
    element_info["ionisation_energy"] = ionisation_energy_info["ionisation_energy"]
    element_info["IR"] = IR_DP_VI["IR"]
    element_info["DP"] = IR_DP_VI["DP"]
    element_info["VI"] = IR_DP_VI["VI"]
    element_info = pd.merge(element_info, bond_info, on=["index", "element"], how="left")
    return element_info

from glob import glob
import os
def get_info(input: str, output: str):
    """
    input:Contains all *.cif structures to read, where "*" must be "formula_space_group_number", e.g.(Ba0.1Sr0.1Ca0.8)HfO3_62.cif, Sr2LaTaO6_160.cifetc.
    """
    element_info_files = glob(f"{input}/*.cif")
    erro_result = pd.DataFrame()
    for file in element_info_files:
        name = os.path.basename(file).split(".cif")[0]
        try:
            structure = Structure.from_file(file)
            formula = name.split("_")[0]
            spn = int(name.split("_")[1])
            element_info = get_element_info(structure, formula = formula, spn = spn)
            element_info.to_csv(f"{output}/{name}.csv", index=False)
        except Exception as e:
            erro_result = erro_result._append({"error": name}, ignore_index=True)
            print(f"Error processing {file}: {e}")
    erro_result.to_csv(f"{output}/error_files.csv", index=False)