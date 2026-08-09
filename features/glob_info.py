import pandas as pd
import math
import os
from pymatgen.core import Structure
from features.crystal import analyze_crystal_data
from features.get_info import get_element_info
from features.lattice_energy import LatticeEnergyCalculator
from features.bond_energy import get_total_bond_energy
import warnings
warnings.simplefilter("ignore")


import re

def parse_chemical_formula(s):
    if pd.isna(s) or s is None:
        return {}
    s = str(s).replace(" ", "")
    pattern = r"([A-Z][a-z]?)([\d\.]+)?"
    matches = re.findall(pattern, s)
    formula_dict = {}
    for elem, num in matches:
        formula_dict[elem] = float(num) if num else 1.0
    return formula_dict

def is_same_formula(a, b):
    """
    Check if two formulas are equivalent (compare with 4 decimal places)
    """
    dict_a = parse_chemical_formula(a)
    dict_b = parse_chemical_formula(b)

    # Return False directly if element types differ
    if set(dict_a.keys()) != set(dict_b.keys()):
        return False

    # Round to 4 decimal places uniformly before comparison
    for elem in dict_a:
        val_a = round(dict_a[elem], 4)
        val_b = round(dict_b[elem], 4)
        if val_a != val_b:
            return False
            
    return True

def formula_standard(string):
    if ":" in string:
        string = string.replace(":","")
    if ", " in string:
        string = string.replace(", ","")
    if "," in string:
        string = string.replace(",","")
    return string
def get_info(datas, formula, read_already, output_dir):
    """
    datas: Folder containing all cif structures
    formula: Table containing formula, space group, A\B1\B2 elements
    """
    rrrrrr = pd.DataFrame()
    read_already_li = read_already["name"].tolist()
    #try:
    for data in datas:
        name = os.path.basename(data)[:-4]
        formu = name.split("_")[0]
        spn = int(name.split("_")[1])
        if name in read_already_li:
            continue
        else:
            # Crystal structure
            structure = Structure.from_file(data)
            info = get_element_info(structure, formu, spn)
            info["element"] = info["element"].apply(formula_standard)
            result = pd.DataFrame()
            result_a = pd.DataFrame()
            file = info
            file_a = file[file["element"] == "O"]
            file_c = file[file["element"] != "O"]
            # Element
            result['element'] = file_c['element']
            # Cation coordination number
            result['coordination_number'] = file_c['coordination_number']
            # Cation mass average
            result['mass_cation'] = file_c.groupby('element')['mass'].transform('mean')
            # Cation covalent radius average
            result['covalent_radius_cation'] = file_c.groupby('element')['covalent_radius'].transform('mean')
            # Cation MB electronegativity average
            result['MB_Electronegativity_cation'] = file_c.groupby('element')['MB_Electronegativity'].transform('mean')
            # Cation valence electronegativity average
            result['valence_Eletronegativity_cation'] = file_c.groupby('element')['valence_Eletronegativity'].transform('mean')
            # CationPaulingElectronegativityaverageValue
            result['Pauling_Electronegativity_cation'] = file_c.groupby('element')['Pauling_Electronegativity'].transform('mean')
            # Cation atomic radius average
            result['atomic_radius_cation'] = file_c.groupby('element')['atomic_radius'].transform('mean')
            # Cation radius average
            result['ionic_radius_cation'] = file_c.groupby('element')['IR'].transform('mean')
            # Cation atomic number average
            result['atomic_number_cation'] = file_c.groupby('element')['atomic_number'].transform('mean')
            # Cation ionization energy average
            result['ionisation_energy_cation'] = file_c.groupby('element')['ionisation_energy'].transform('mean')
            # Cation covalent volume
            result["convalent_volume_cation"] = file_c.groupby('element')['convalent_volume'].transform('mean')
            # Cation atomic volume
            result["atomic_volume_cation"] = file_c.groupby('element')['atomic_volume'].transform('mean')
            # Cation ionic volume
            result["ionic_volume_cation"] = file_c.groupby('element')['VI'].transform('mean')
            # Cation dielectric polarizability
            result["dielectric_polarizability_cation"] = file_c.groupby('element')['DP'].transform('mean')
            # Bond length average
            result['mean_bond_length'] = file_c.groupby('element')['avg_bond_length'].transform('mean')
            # Bond length standard deviation
            result['std_bond_length'] = file_c.groupby('element')['std_bond_length'].transform('mean')
            # Cation bond valence sum
            result['bvs_cation'] = file_c.groupby('element')['bvs'].transform('mean')
            # Bond strength average
            result['mean_bond_strength'] = file_c.groupby('element')['avg_bond_strength'].transform('mean')
            # Bond strength standard deviation
            result['std_bond_strength'] = file_c.groupby('element')['std_bond_strength'].transform('mean')
            # CovalenceaverageValue
            result['mean_covalence'] = file_c.groupby('element')['avg_covalence'].transform('mean')
            # CovalenceStandardPoor
            result['std_covalence'] = file_c.groupby('element')['std_covalence'].transform('mean')
            # CovalenceDegreeaverageValue
            result['mean_cvd'] = file_c.groupby('element')['avg_cvd'].transform('mean')
            # CovalenceDegreeStandardPoor
            result['std_cvd'] = file_c.groupby('element')['std_cvd'].transform('mean')
            # IonicityaverageValue
            result['avg_fi'] = file_c.groupby('element')['avg_fi'].transform('mean')
            # Lattice energyaverageValue
            result['avg_U'] = file_c.groupby('element')['avg_U'].transform('mean')
            # Bond energyaverageValue
            result['avg_E'] = file_c.groupby('element')['avg_E'].transform('mean')

            result = result.drop_duplicates(subset=['element'])
            result = result.reset_index(drop=True)
            # AnionElement
            result_a['element'] = file_a['element']
            # AnionQualityQuantityaverageValue
            result_a['mass_anion'] = file_a.groupby('element')['mass'].transform('mean')
            # Anion covalent radiusaverageValue
            result_a['covalent_radius_anion'] = file_a.groupby('element')['covalent_radius'].transform('mean')
            # AnionMBElectronegativityaverageValue
            result_a['MB_Electronegativity_anion'] = file_a.groupby('element')['MB_Electronegativity'].transform('mean')
            # AnionValence electronegativityaverageValue
            result_a['valence_Eletronegativity_anion'] = file_a.groupby('element')['valence_Eletronegativity'].transform('mean')
            # AnionPaulingElectronegativityaverageValue
            result_a['Pauling_Electronegativity_anion'] = file_a.groupby('element')['Pauling_Electronegativity'].transform('mean')
            # Anion atomic radiusaverageValue
            result_a['atomic_radius_anion'] = file_a.groupby('element')['atomic_radius'].transform('mean')
            # AnionRadiusaverageValue
            result_a['ionic_radius_anion'] = file_a.groupby('element')['IR'].transform('mean')
            # AnionAtomSequenceCountaverageValue
            result_a['atomic_number_anion'] = file_a.groupby('element')['atomic_number'].transform('mean')
            # Anion ionization energyaverageValue
            result_a['ionisation_energy_anion'] = file_a.groupby('element')['ionisation_energy'].transform('mean')
            # AnionTotalValence volume
            result_a["convalent_volume_anion"] = file_a.groupby('element')['convalent_volume'].transform('mean')
            # AnionAtomic volume
            result_a["atomic_volume_anion"] = file_a.groupby('element')['atomic_volume'].transform('mean')
            # AnionIonic volume
            result_a["ionic_volume_anion"] = file_a.groupby('element')['VI'].transform('mean')
            # AnionDielectric polarizability
            result_a["dielectric_polarizability_anion"] = file_a.groupby('element')['DP'].transform('mean')
            # Cation bond valence sum
            result_a['bvs_anion'] = file_a.groupby('element')['bvs'].transform('mean')
            result_a = result_a.drop_duplicates(subset=['element'])
            result_a = result_a.reset_index(drop=True)
            # Crystal structureInformation
            ac_list = analyze_crystal_data(structure, formu, spn)
            # Space group
            sp = ac_list["sp"]
            # Crystal system
            crystal_system = ac_list["crystal_system"]
            # Theoretical density
            density = ac_list["density"]
            # LatticeParameters
            a = ac_list["a"]
            b = ac_list["b"]
            c = ac_list["c"]
            alpha = ac_list["alpha"]
            beta = ac_list["beta"]
            gamma = ac_list["gamma"]
            lsd = ac_list["lsd"]
            asd = ac_list["asd"]
            # Unit cell volume
            volume = ac_list["volume"]
            # GlobalNotStability index
            GII = ac_list["GII"]
            # Bond valence
            bvsc = ac_list["bvsc"]
            bvsa = ac_list["bvsa"]
            # Lattice energy
            lattice_energy_calculator = LatticeEnergyCalculator(structure, formu, spn)
            U = lattice_energy_calculator.get_lattice_energy_info()[0]
            E = get_total_bond_energy(structure, formu, spn)[1]
            lattice_energy = lattice_energy_calculator.get_lattice_energy_info()[1]
            bond_energy = get_total_bond_energy(structure, formu, spn)[0]
            if "O3" in name:
                print(name)
                # Site
                # A = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["A"].values[0]]
                target_A = formula[(formula["name"] == name) & (formula["sp"] == spn)]["A"].values[0]
                A = result[result["element"].apply(lambda x: is_same_formula(x, target_A))]
                A = A.reset_index(drop=True)

                # B1 = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["B1"].values[0]]
                target_B1 = formula[(formula["name"] == name) & (formula["sp"] == spn)]["B1"].values[0]
                B1 = result[result["element"].apply(lambda x: is_same_formula(x, target_B1))]
                B1 = B1.reset_index(drop=True)

                O = result_a
                O = O.reset_index(drop=True)
                # Ionic volume
                vi = A["ionic_volume_cation"][0] + B1["ionic_volume_cation"][0] + O["ionic_volume_anion"][0] * 3
                # Atomic volume
                va = A["atomic_volume_cation"][0] + B1["atomic_volume_cation"][0] + O["atomic_volume_anion"][0] * 3
                # TotalValence volume
                vc = A["convalent_volume_cation"][0] + B1["convalent_volume_cation"][0] + O["convalent_volume_anion"][0] * 3
                # PhaseCorrectMolecular weightQuantity
                m = A["mass_cation"][0] + B1["mass_cation"][0] + O["mass_anion"][0] * 3
                # Ionic packing fractionCount
                pfi = vi / volume
                # AtomPacking fractionCount
                pfa = vi / volume
                # TotalValence packing fractionCount
                pfc = vc / volume
                # Molecular dielectric polarizability
                p = A["dielectric_polarizability_cation"][0] + B1["dielectric_polarizability_cation"][0] + O["dielectric_polarizability_anion"][0] * 3
                # UnitVolume molecular dielectric polarizability
                ppv = p / volume
                # Based on atomic radius
                t = (A["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / (math.sqrt(2) * (B1["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]))
                # BaseAtKeyLong
                tt = A['mean_bond_length'][0] / (math.sqrt(2) * B1['mean_bond_length'][0])
                # Octahedral factor
                of = A["atomic_radius_cation"][0] / B1["atomic_radius_cation"][0]
                # AtomSequenceCountaverageValue
                an = (A["atomic_number_cation"][0] + B1["atomic_number_cation"][0]) / 2 
                # Cation covalent radius
                crc = (A["covalent_radius_cation"][0] + B1["covalent_radius_cation"][0]) / 2
                # CationAtomRadius
                arc = (A["atomic_radius_cation"][0] + B1["atomic_radius_cation"][0]) / 2
                # CationIonic radius
                irc = (A['ionic_radius_cation'][0] + B1['ionic_radius_cation'][0]) / 2
                # CationQualityQuantity
                mc = (A["mass_cation"][0] + B1["mass_cation"][0]) / 2
                # Cation polarizability
                pc = (A["dielectric_polarizability_cation"][0] + B1["dielectric_polarizability_cation"][0]) / 2
                # CationPaulingElectronegativity
                enPc = (A["Pauling_Electronegativity_cation"][0] + B1["Pauling_Electronegativity_cation"][0]) / 2
                # CationMBElectronegativity
                enMBc = (A["MB_Electronegativity_cation"][0] + B1["MB_Electronegativity_cation"][0]) / 2
                # CationValence electronegativity
                envc = (A["valence_Eletronegativity_cation"][0] + B1["valence_Eletronegativity_cation"][0]) / 2
                # CationIonization energy
                iec = (A["ionisation_energy_cation"][0] + B1["ionisation_energy_cation"][0]) / 2
                # Cation atomic volumeaverageValue
                vac = (A["atomic_volume_cation"][0] + B1["atomic_volume_cation"][0]) / 2
                # Cation covalent volumeaverageValue
                vcc = (A["convalent_volume_cation"][0] + B1["convalent_volume_cation"][0]) / 2
                # Cation ionic volumeaverageValue
                vic = (A["ionic_volume_cation"][0] + B1["ionic_volume_cation"][0]) / 2
                # Anion covalent radius
                cra = O['covalent_radius_anion'][0]
                # AnionAtomRadius
                ara = O['atomic_radius_anion'][0]
                # AnionIonic radius
                ira = O['ionic_radius_anion'][0]
                # AnionQualityQuantity
                ma = O["mass_anion"][0]
                # AnionPolarizability
                pa = O["dielectric_polarizability_anion"][0]
                # AnionPaulingElectronegativity
                enPa = O['Pauling_Electronegativity_anion'][0]
                # AnionMBElectronegativity
                enMBa = O['MB_Electronegativity_anion'][0]
                # AnionValence electronegativity
                enva = O['valence_Eletronegativity_anion'][0]
                # AnionIonization energy
                iea = O['ionisation_energy_anion'][0]
                # AnionAtomic volumeaverageValue
                vaa = O["atomic_volume_anion"][0]
                # AnionTotalValence volumeaverageValue
                vca = O["convalent_volume_anion"][0]
                # AnionIonic volumeaverageValue
                via = O["ionic_volume_anion"][0]
                # Bond length average
                blm = (A["mean_bond_length"][0] + B1["mean_bond_length"][0]) / 2
                # Bond length standard deviation
                blsd = (A["std_bond_length"][0] + B1["std_bond_length"][0]) / 2
                # Bond strengthsum
                S = A['coordination_number'][0] * B1["mean_bond_strength"][0] + A['coordination_number'][0] * B1["mean_bond_strength"][0]
                # Bond strength average
                Sm = (A["mean_bond_strength"][0] + B1["mean_bond_strength"][0]) / 2
                # Bond strength standard deviation
                Ssd = (A["std_bond_strength"][0] + B1["std_bond_strength"][0]) / 2
                # CovalenceaverageValue
                cvm = (A["mean_covalence"][0] + B1["mean_covalence"][0] ) / 2
                # CovalenceStandardPoor
                cvsd = (A["std_covalence"][0] + B1["std_covalence"][0]) / 2
                # CovalenceDegreeaverageValue
                cvd = (A["mean_cvd"][0] + B1["mean_cvd"][0]) / 2
                # CovalenceDegreeStandardPoor
                cvdsd = (A["std_cvd"][0] + B1["std_cvd"][0]) / 2
                # Ionicity degreeaverageValue
                fi = (A["avg_fi"][0] + B1["avg_fi"][0]) / 2
                # Lattice energyaverageValue
                Um = (A["avg_U"][0] + B1["avg_U"][0]) / 2
                # Bond energyaverageValue
                Em = (A["avg_E"][0] + B1["avg_E"][0]) / 2

                rrrrrr = rrrrrr._append(
                    {
                        "name":name, "sp": sp, "crystal_system": crystal_system, "density": density, "volume": volume,
                        "a": a, "b":b, "c":c,  "alpha":alpha, "beta":beta, "gamma":gamma, "asd":asd, "lsd": lsd, "GII": GII,
                        "vi": vi, "va": va, "vc": vc, "m": m, "pfi": pfi, "pfa": pfa, "pfc": pfc, "p": p, "ppv": ppv,
                        "t": t, "tt": tt, "of": of, "an": an, "crc": crc, "arc": arc, "irc": irc, "mc": mc, "pc": pc, 
                        "enPc": enPc, "enMBc": enMBc, "envc": envc, "iec": iec, "vac": vac, "vcc": vcc, "vic": vic, "cra": cra,
                        "ara": ara, "ira": ira, "ma": ma, "pa": pa, "enPa": enPa, "enMBa": enMBa, "enva": enva, "iea": iea,
                        "vaa": vaa, "vca": vca, "via": via, "blm": blm, "blsd": blsd, "S": S, "Sm": Sm, "Ssd": Ssd, "cvm": cvm,
                        "cvsd": cvsd, "cvd":cvd, "cvdsd": cvdsd,  "bvsc": bvsc, "bvsa":bvsa, 
                        "lattice_energy": lattice_energy, "bond_energy": bond_energy,"U": U, "E": E, "fi": fi, "Um": Um, "Em": Em,
                        "A": A["element"][0], "cnA": A['coordination_number'][0], 
                        "B1": B1["element"][0], "cnB1": B1['coordination_number'][0],
                        "B2": None, "cnB2": None,
                    }, ignore_index = True
                )
                rrrrrr.to_csv(output_dir, index = False)
            elif "O6" in name:
                print(name)
                # Site
                # A = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["A"].values[0]]
                target_A = formula[(formula["name"] == name) & (formula["sp"] == spn)]["A"].values[0]
                A = result[result["element"].apply(lambda x: is_same_formula(x, target_A))]
                A = A.reset_index(drop=True)

                # B1 = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["B1"].values[0]]
                target_B1 = formula[(formula["name"] == name) & (formula["sp"] == spn)]["B1"].values[0]
                B1 = result[result["element"].apply(lambda x: is_same_formula(x, target_B1))]
                B1 = B1.reset_index(drop=True)

                # B2 = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["B2"].values[0]]
                target_B2 = formula[(formula["name"] == name) & (formula["sp"] == spn)]["B2"].values[0]
                B2 = result[result["element"].apply(lambda x: is_same_formula(x, target_B2))]
                B2 = B2.reset_index(drop=True)

                O = result_a
                O = O.reset_index(drop=True)
                # Ionic volume
                vi = A["ionic_volume_cation"][0] * 2 + B1["ionic_volume_cation"][0]+ B2["ionic_volume_cation"][0] + O["ionic_volume_anion"][0] * 6
                # Atomic volume
                va = A["atomic_volume_cation"][0] * 2 + B1["atomic_volume_cation"][0] + B2["atomic_volume_cation"][0] + O["atomic_volume_anion"][0] * 6
                # TotalValence volume
                vc = A["convalent_volume_cation"][0] * 2 + B1["convalent_volume_cation"][0] + B2["convalent_volume_cation"][0] + O["convalent_volume_anion"][0] * 6
                # PhaseCorrectMolecular weightQuantity
                m = A["mass_cation"][0] * 2 + B1["mass_cation"][0] + B2["mass_cation"][0] + O["mass_anion"][0] * 6
                # Ionic packing fractionCount
                pfi = vi / volume
                # AtomPacking fractionCount
                pfa = vi / volume
                # TotalValence packing fractionCount
                pfc = vc / volume
                # Molecular dielectric polarizability
                p = A["dielectric_polarizability_cation"][0] * 2 + B1["dielectric_polarizability_cation"][0] + B2["dielectric_polarizability_cation"][0] + O["dielectric_polarizability_anion"][0] * 6
                # UnitVolume molecular dielectric polarizability
                ppv = p / volume
                # Based on atomic radius
                t = (A["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / (math.sqrt(2) * (B1["atomic_radius_cation"][0] + O['atomic_radius_anion'][0] + B2["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / 2)
                # BaseAtKeyLong
                tt = A['mean_bond_length'][0] / (math.sqrt(2) * (B1['mean_bond_length'][0] + B2['mean_bond_length'][0]) / 2)
                # Octahedral factor
                of = A["atomic_radius_cation"][0] / ((B1["atomic_radius_cation"][0] + B2["atomic_radius_cation"][0]) / 2)
                # AtomSequenceCountaverageValue
                an = (A["atomic_number_cation"][0] + (B1["atomic_number_cation"][0] + B2["atomic_number_cation"][0]) / 2) / 2
                # Cation covalent radius
                crc = (A["covalent_radius_cation"][0] + (B1["covalent_radius_cation"][0] + B2["covalent_radius_cation"][0]) / 2) / 2
                # CationAtomRadius
                arc = (A["atomic_radius_cation"][0] + (B1["atomic_radius_cation"][0] + B2["atomic_radius_cation"][0]) / 2) / 2
                # CationIonic radius
                irc = (A['ionic_radius_cation'][0] + (B1['ionic_radius_cation'][0] + B2['ionic_radius_cation'][0]) / 2) / 2
                # CationQualityQuantity
                mc = (A["mass_cation"][0] + (B1["mass_cation"][0] + B2["mass_cation"][0]) / 2) / 2
                # Cation polarizability
                pc = (A["dielectric_polarizability_cation"][0] + (B1["dielectric_polarizability_cation"][0] + B2["dielectric_polarizability_cation"][0]) / 2) / 2
                # CationPaulingElectronegativity
                enPc = (A["Pauling_Electronegativity_cation"][0] + (B1["Pauling_Electronegativity_cation"][0] + B2["Pauling_Electronegativity_cation"][0]) / 2) / 2
                # CationMBElectronegativity
                enMBc = (A["MB_Electronegativity_cation"][0] + (B1["MB_Electronegativity_cation"][0] + B2["MB_Electronegativity_cation"][0]) / 2) / 2
                # CationValence electronegativity
                envc = (A["valence_Eletronegativity_cation"][0] + (B1["valence_Eletronegativity_cation"][0] + B2["valence_Eletronegativity_cation"][0]) / 2) / 2
                # CationIonization energy
                iec = (A["ionisation_energy_cation"][0] + (B1["ionisation_energy_cation"][0] + B2["ionisation_energy_cation"][0]) / 2) / 2
                # Cation atomic volumeaverageValue
                vac = (A["atomic_volume_cation"][0] + (B1["atomic_volume_cation"][0] + B2["atomic_volume_cation"][0]) / 2) / 2
                # Cation covalent volumeaverageValue
                vcc = (A["convalent_volume_cation"][0] + (B1["convalent_volume_cation"][0] + B2["convalent_volume_cation"][0]) / 2) / 2
                # Cation ionic volumeaverageValue
                vic = (A["ionic_volume_cation"][0] + (B1["ionic_volume_cation"][0] + B2["ionic_volume_cation"][0]) / 2) / 2
                # Anion covalent radius
                cra = O['covalent_radius_anion'][0]
                # AnionAtomRadius
                ara = O['atomic_radius_anion'][0]
                # AnionIonic radius
                ira = O['ionic_radius_anion'][0]
                # AnionQualityQuantity
                ma = O["mass_anion"][0]
                # AnionPolarizability
                pa = O["dielectric_polarizability_anion"][0]
                # AnionPaulingElectronegativity
                enPa = O['Pauling_Electronegativity_anion'][0]
                # AnionMBElectronegativity
                enMBa = O['MB_Electronegativity_anion'][0]
                # AnionValence electronegativity
                enva = O['valence_Eletronegativity_anion'][0]
                # AnionIonization energy
                iea = O['ionisation_energy_anion'][0]
                # AnionAtomic volumeaverageValue
                vaa = O["atomic_volume_anion"][0]
                # AnionTotalValence volumeaverageValue
                vca = O["convalent_volume_anion"][0]
                # AnionIonic volumeaverageValue
                via = O["ionic_volume_anion"][0]
                # Bond length average
                blm = (A["mean_bond_length"][0] + (B1["mean_bond_length"][0] + B2["mean_bond_length"][0]) / 2) / 2
                # Bond length standard deviation
                blsd = (A["std_bond_length"][0] + (B1["std_bond_length"][0] + B2["std_bond_length"][0]) / 2) / 2
                # Bond strengthsum
                S = A['coordination_number'][0] * A["mean_bond_strength"][0] * 2 + B1['coordination_number'][0] * B1["mean_bond_strength"][0] + B2['coordination_number'][0] * B2["mean_bond_strength"][0]
                # Bond strength average
                Sm = (A["mean_bond_strength"][0] + (B1["mean_bond_strength"][0] + B2["mean_bond_strength"][0]) / 2) / 2
                # Bond strength standard deviation
                Ssd = (A["std_bond_strength"][0] + (B1["std_bond_strength"][0] + B2["std_bond_strength"][0]) / 2) / 2
                # CovalenceaverageValue
                cvm = (A["mean_covalence"][0] + (B1["mean_covalence"][0] + B2["mean_covalence"][0]) / 2 ) / 2
                # CovalenceStandardPoor
                cvsd = (A["std_covalence"][0] + (B1["std_covalence"][0] + B2["std_covalence"][0]) / 2) / 2
                # CovalenceDegreeaverageValue
                cvd = (A["mean_cvd"][0] + (B1["mean_cvd"][0] + B2["mean_cvd"][0]) / 2) / 2
                # CovalenceDegreeStandardPoor
                cvdsd = (A["std_cvd"][0] + (B1["std_cvd"][0] + B2["std_cvd"][0]) / 2) / 2
                # Ionicity degreeaverageValue
                fi = (A["avg_fi"][0] + (B1["avg_fi"][0] + B2["avg_fi"][0]) / 2) / 2
                # Lattice energyaverageValue
                Um = (A["avg_U"][0] + (B1["avg_U"][0] + B2["avg_U"][0]) / 2) / 2
                # Bond energyaverageValue
                Em = (A["avg_E"][0] + (B1["avg_E"][0] + B2["avg_E"][0]) / 2) / 2

                rrrrrr = rrrrrr._append(
                    {
                        "name":name, "sp": sp, "crystal_system": crystal_system, "density": density, "volume": volume,
                        "a": a, "b":b, "c":c,  "alpha":alpha, "beta":beta, "gamma":gamma, "asd":asd, "lsd": lsd, "GII": GII,
                        "vi": vi, "va": va, "vc": vc, "m": m, "pfi": pfi, "pfa": pfa, "pfc": pfc, "p": p, "ppv": ppv,
                        "t": t, "tt": tt, "of": of, "an": an, "crc": crc, "arc": arc, "irc": irc, "mc": mc, "pc": pc, 
                        "enPc": enPc, "enMBc": enMBc, "envc": envc, "iec": iec, "vac": vac, "vcc": vcc, "vic": vic, "cra": cra,
                        "ara": ara, "ira": ira, "ma": ma, "pa": pa, "enPa": enPa, "enMBa": enMBa, "enva": enva, "iea": iea,
                        "vaa": vaa, "vca": vca, "via": via, "blm": blm, "blsd": blsd, "S": S, "Sm": Sm, "Ssd": Ssd, "cvm": cvm,
                        "cvsd": cvsd, "cvd":cvd, "cvdsd": cvdsd,  "bvsc": bvsc, "bvsa":bvsa, 
                        "lattice_energy": lattice_energy, "bond_energy": bond_energy,"U": U, "E": E, "fi": fi, "Um": Um, "Em": Em,
                        "A": A["element"][0], "cnA": A['coordination_number'][0], 
                        "B1": B1["element"][0], "cnB1": B1['coordination_number'][0],
                        "B2": B2["element"][0], "cnB2": B2['coordination_number'][0],
                    }, ignore_index = True
                )
                rrrrrr.to_csv(output_dir, index = False)
            elif "O9" in name:
                print(name)
                # Site
                #A = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["A"].values[0]]
                target_A = formula[(formula["name"] == name) & (formula["sp"] == spn)]["A"].values[0]
                A = result[result["element"].apply(lambda x: is_same_formula(x, target_A))]
                A = A.reset_index(drop=True)

                # B1 = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["B1"].values[0]]
                target_B1 = formula[(formula["name"] == name) & (formula["sp"] == spn)]["B1"].values[0]
                B1 = result[result["element"].apply(lambda x: is_same_formula(x, target_B1))]
                B1 = B1.reset_index(drop=True)

                #B2 = result[result["element"] == formula[(formula["name"] == formu) & (formula["sp"] == spn)]["B2"].values[0]]
                target_B2 = formula[(formula["name"] == name) & (formula["sp"] == spn)]["B2"].values[0]
                B2 = result[result["element"].apply(lambda x: is_same_formula(x, target_B2))]
                B2 = B2.reset_index(drop=True)
                
                O = result_a
                O = O.reset_index(drop=True)
                # Ionic volume
                vi = A["ionic_volume_cation"][0] * 3 + B1["ionic_volume_cation"][0] + B2["ionic_volume_cation"][0] * 2 + O["ionic_volume_anion"][0] * 9
                # Atomic volume
                va = A["atomic_volume_cation"][0] * 3 + B1["atomic_volume_cation"][0] + B2["atomic_volume_cation"][0] * 2 + O["atomic_volume_anion"][0] * 9
                # TotalValence volume
                vc = A["convalent_volume_cation"][0] * 3 + B1["convalent_volume_cation"][0] + B2["convalent_volume_cation"][0] * 2 + O["convalent_volume_anion"][0] * 9
                # PhaseCorrectMolecular weightQuantity
                m = A["mass_cation"][0] * 3 + B1["mass_cation"][0] + B2["mass_cation"][0] * 2 + O["mass_anion"][0] * 9
                # Ionic packing fractionCount
                pfi = vi / volume
                # AtomPacking fractionCount
                pfa = vi / volume
                # TotalValence packing fractionCount
                pfc = vc / volume
                # Molecular dielectric polarizability
                p = A["dielectric_polarizability_cation"][0] * 3 + B1["dielectric_polarizability_cation"][0] + B2["dielectric_polarizability_cation"][0] * 2 + O["dielectric_polarizability_anion"][0] * 9
                # UnitVolume molecular dielectric polarizability
                ppv = p / volume
                # Based on atomic radius
                t = (A["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / (math.sqrt(2) * ((B1["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / 3 + 2 * (B2["atomic_radius_cation"][0] + O['atomic_radius_anion'][0]) / 3))
                # BaseAtKeyLong
                tt = A['mean_bond_length'][0] / (math.sqrt(2) * (B1['mean_bond_length'][0] / 3 + 2 * B2['mean_bond_length'][0] / 3))
                # Octahedral factor
                of = A["atomic_radius_cation"][0] / (B1["atomic_radius_cation"][0] / 3 + 2 * B2["atomic_radius_cation"][0] / 3)
                # AtomSequenceCountaverageValue
                an = (A["atomic_number_cation"][0] + (B1["atomic_number_cation"][0] + 2 * B2["atomic_number_cation"][0]) / 3) / 2
                # Cation covalent radius
                crc = (A["covalent_radius_cation"][0] + (B1["covalent_radius_cation"][0] + 2 * B2["covalent_radius_cation"][0]) / 3) / 2
                # CationAtomRadius
                arc = (A["atomic_radius_cation"][0] + (B1["atomic_radius_cation"][0] + 2 * B2["atomic_radius_cation"][0]) / 3) / 2
                # CationIonic radius
                irc = (A['ionic_radius_cation'][0] + (B1['ionic_radius_cation'][0] + 2 * B2['ionic_radius_cation'][0]) / 3) / 2
                # CationQualityQuantity
                mc = (A["mass_cation"][0] + (B1["mass_cation"][0] + 2 * B2["mass_cation"][0]) / 3) / 2
                # Cation polarizability
                pc = (A["dielectric_polarizability_cation"][0] + (B1["dielectric_polarizability_cation"][0] + 2 * B2["dielectric_polarizability_cation"][0]) / 3) / 2
                # CationPaulingElectronegativity
                enPc = (A["Pauling_Electronegativity_cation"][0] + (B1["Pauling_Electronegativity_cation"][0] + 2 * B2["Pauling_Electronegativity_cation"][0]) / 3) / 2
                # CationMBElectronegativity
                enMBc = (A["MB_Electronegativity_cation"][0] + (B1["MB_Electronegativity_cation"][0] + 2 * B2["MB_Electronegativity_cation"][0]) / 3) / 2
                # CationValence electronegativity
                envc = (A["valence_Eletronegativity_cation"][0] + (B1["valence_Eletronegativity_cation"][0] + 2 * B2["valence_Eletronegativity_cation"][0]) / 3) / 2
                # CationIonization energy
                iec = (A["ionisation_energy_cation"][0] + (B1["ionisation_energy_cation"][0] + 2 * B2["ionisation_energy_cation"][0]) / 3) / 2
                # Cation atomic volumeaverageValue
                vac = (A["atomic_volume_cation"][0] + (B1["atomic_volume_cation"][0] + 2 * B2["atomic_volume_cation"][0]) / 3) / 2
                # Cation covalent volumeaverageValue
                vcc = (A["convalent_volume_cation"][0] + (B1["convalent_volume_cation"][0] + 2 * B2["convalent_volume_cation"][0]) / 3) / 2
                # Cation ionic volumeaverageValue
                vic = (A["ionic_volume_cation"][0] + (B1["ionic_volume_cation"][0] + 2 * B2["ionic_volume_cation"][0]) / 3) / 2
                # Anion covalent radius
                cra = O['covalent_radius_anion'][0]
                # AnionAtomRadius
                ara = O['atomic_radius_anion'][0]
                # AnionIonic radius
                ira = O['ionic_radius_anion'][0]
                # AnionQualityQuantity
                ma = O["mass_anion"][0]
                # AnionPolarizability
                pa = O["dielectric_polarizability_anion"][0]
                # AnionPaulingElectronegativity
                enPa = O['Pauling_Electronegativity_anion'][0]
                # AnionMBElectronegativity
                enMBa = O['MB_Electronegativity_anion'][0]
                # AnionValence electronegativity
                enva = O['valence_Eletronegativity_anion'][0]
                # AnionIonization energy
                iea = O['ionisation_energy_anion'][0]
                # AnionAtomic volumeaverageValue
                vaa = O["atomic_volume_anion"][0]
                # AnionTotalValence volumeaverageValue
                vca = O["convalent_volume_anion"][0]
                # AnionIonic volumeaverageValue
                via = O["ionic_volume_anion"][0]
                # Bond length average
                blm = (A["mean_bond_length"][0] + (B1["mean_bond_length"][0] + 2 * B2["mean_bond_length"][0]) / 3) / 2
                # Bond length standard deviation
                blsd = (A["std_bond_length"][0] + (B1["std_bond_length"][0] + 2 * B2["std_bond_length"][0]) / 3) / 2
                # Bond strengthsum
                S = A['coordination_number'][0] * A["mean_bond_strength"][0] * 3 + B1['coordination_number'][0] * B1["mean_bond_strength"][0] + B2['coordination_number'][0] * B2["mean_bond_strength"][0] * 2
                # Bond strength average
                Sm = (A["mean_bond_strength"][0] + (B1["mean_bond_strength"][0] + 2 * B2["mean_bond_strength"][0]) / 3) / 2
                # Bond strength standard deviation
                Ssd = (A["std_bond_strength"][0] + (B1["std_bond_strength"][0] + 2 * B2["std_bond_strength"][0]) / 3) / 2
                # CovalenceaverageValue
                cvm = (A["mean_covalence"][0] + (B1["mean_covalence"][0] + 2 * B2["mean_covalence"][0]) / 3 ) / 2
                # CovalenceStandardPoor
                cvsd = (A["std_covalence"][0] + (B1["std_covalence"][0] + 2 * B2["std_covalence"][0]) / 3) / 2
                # CovalenceDegreeaverageValue
                cvd = (A["mean_cvd"][0] + (B1["mean_cvd"][0] + 2 * B2["mean_cvd"][0]) / 3) / 2
                # CovalenceDegreeStandardPoor
                cvdsd = (A["std_cvd"][0] + (B1["std_cvd"][0] + 2 * B2["std_cvd"][0]) / 3) / 2
                # Ionicity degreeaverageValue
                fi = (A["avg_fi"][0] + (B1["avg_fi"][0] + 2 * B2["avg_fi"][0]) / 3) / 2
                # Lattice energyaverageValue
                Um = (A["avg_U"][0] + (B1["avg_U"][0] + 2 * B2["avg_U"][0]) / 3) / 2
                # Bond energyaverageValue
                Em = (A["avg_E"][0] + (B1["avg_E"][0] + 2 * B2["avg_E"][0]) / 3) / 2
                
                rrrrrr = rrrrrr._append(
                    {
                        "name":name, "sp": sp, "crystal_system": crystal_system, "density": density, "volume": volume,
                        "a": a, "b":b, "c":c,  "alpha":alpha, "beta":beta, "gamma":gamma, "asd":asd, "lsd": lsd, "GII": GII,
                        "vi": vi, "va": va, "vc": vc, "m": m, "pfi": pfi, "pfa": pfa, "pfc": pfc, "p": p, "ppv": ppv,
                        "t": t, "tt": tt, "of": of, "an": an, "crc": crc, "arc": arc, "irc": irc, "mc": mc, "pc": pc, 
                        "enPc": enPc, "enMBc": enMBc, "envc": envc, "iec": iec, "vac": vac, "vcc": vcc, "vic": vic, "cra": cra,
                        "ara": ara, "ira": ira, "ma": ma, "pa": pa, "enPa": enPa, "enMBa": enMBa, "enva": enva, "iea": iea,
                        "vaa": vaa, "vca": vca, "via": via, "blm": blm, "blsd": blsd, "S": S, "Sm": Sm, "Ssd": Ssd, "cvm": cvm,
                        "cvsd": cvsd, "cvd":cvd, "cvdsd": cvdsd,  "bvsc": bvsc, "bvsa":bvsa, 
                        "lattice_energy": lattice_energy, "bond_energy": bond_energy,"U": U, "E": E, "fi": fi, "Um": Um, "Em": Em,
                        "A": A["element"][0], "cnA": A['coordination_number'][0], 
                        "B1": B1["element"][0], "cnB1": B1['coordination_number'][0],
                        "B2": B2["element"][0], "cnB2": B2['coordination_number'][0],
                    }, ignore_index = True
                )
                rrrrrr.to_csv(output_dir, index = False)
            else:
                print(f"{name} NotAttributeAtABO3Perovskitestructures")
    return rrrrrr
    #except:
        #print("OutputCurrentIssue")
        #return rrrrrr
    """
    Unordered formula stringMatchas standard order
    :param messy_str: unordered string(Such as"Ta0.22 Nb0.22Zn0.17Co0.17Sb0.22")
    :param standard_str: StandardOrdered string(Such as"Zn0.17Co0.17Ta0.22Nb0.22Sb0.22")
    :return: StandardBackString
    """
    try:
        # Parse standard stringandunordered string
        standard_dict = parse_chemical_string(standard_str)
        messy_dict = parse_chemical_string(messy_str)
        
        # ValidateElementIsNoOneLead to(AvoidMiss/MultipleElement)
        if set(standard_dict.keys()) != set(messy_dict.keys()):
            raise ValueError(f"ElementNotMatch!StandardElement: {sorted(standard_dict.keys())},UnorderedElement: {sorted(messy_dict.keys())}")
        
        # ByStandardOrderConcatenateString
        normalized_parts = []
        for elem in standard_dict.keys():
            num = messy_dict[elem]
            # HasCountValueThenConcatenate"Element+CountValue",NoneThenOnlyAddElement
            normalized_parts.append(f"{elem}{num}" if num else elem)
        
        return "".join(normalized_parts)
    
    except Exception as e:
        print(f"ProcessingFailed: {e}")
        return None