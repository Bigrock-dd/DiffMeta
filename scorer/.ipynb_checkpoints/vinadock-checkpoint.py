from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import meeko
import warnings
import time
from tqdm import tqdm 
warnings.filterwarnings("ignore")

def vinadock(target, smiles_list):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = target.lower()

    box_dict = {
        '7w4a':  ((97.6, 113.4, 86.9), (30, 30, 30))
    }

    if target not in box_dict:
        raise ValueError(f"[vinadock] Unknown target: {target}")
    box_center, box_size = box_dict[target]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    receptor = os.path.join(script_dir, "receptors", f"{target.upper()}.pdbqt")

    if not os.path.exists(receptor):
        raise FileNotFoundError(f"[vinadock] Receptor file not found: {receptor}")

    print(f"[vinadock] Using receptor: {receptor}")
    print(f"[vinadock] Box center: {box_center}, Box size: {box_size}")

    score_list = []

    for idx, smiles in enumerate(tqdm(smiles_list, desc=f"Docking {target.upper()} ligands")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit failed to parse SMILES.")

            mol = Chem.AddHs(mol)
            status = AllChem.EmbedMolecule(mol)
            if status == -1:
                raise ValueError("RDKit failed to embed 3D.")

            meeko_prep = meeko.MoleculePreparation()
            meeko_prep.prepare(mol)
            lig_pdbqt = meeko_prep.write_pdbqt_string()

            with open(f"ligand_{timestamp}_{idx + 1}.pdbqt", "w") as f:
                f.write(lig_pdbqt)

            v = Vina(sf_name='vina', verbosity=0)
            v.set_receptor(receptor)
            v.set_ligand_from_string(lig_pdbqt)
            v.compute_vina_maps(center=box_center, box_size=box_size)

            v.dock(exhaustiveness=16, n_poses=20)

            energies = v.energies()
#            print(f"[vinadock] Raw energy list: {energies}") 
            if energies is None or len(energies) == 0:
                raise ValueError("Docking failed, no poses found.")

            docking_score = -min(e[0] for e in energies)  
            score_list.append(docking_score)

        except Exception as e:
            print(f"[vinadock] Error docking SMILES #{idx + 1}: {smiles}")
            print(f"[vinadock] Error: {e}")
            score_list.append(-1)

    return score_list