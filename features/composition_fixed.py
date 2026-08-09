from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.composition import Species
# Override Site class species_string property to return string containing element symbol and occupancy with 4 decimal places
"""
@property
def fixed_species_string(self):
    parts = []
    for sp, occu in sorted(self.species.items(), key=lambda x: x[0].X):
        if isinstance(sp, (Species, DummySpecies)):
            parts.append(f"{sp}:{occu:.4f}") 
        else:
            parts.append(f"{sp}:{occu:.4f}")
    return ", ".join(parts)
"""
@property
def fixed_species_string(self):
    # Only 1 element and occupancy=1 → show element symbol only
    if len(self.species) == 1 and next(iter(self.species.values())) == 1.0:
        return next(iter(self.species.keys())).symbol
    # Solid solution / occupancy <1 → show element:4 decimal places
    parts = []
    for elem, occ in self.species.items():
        parts.append(f"{elem.symbol}:{occ:.4f}")
    return ", ".join(parts)