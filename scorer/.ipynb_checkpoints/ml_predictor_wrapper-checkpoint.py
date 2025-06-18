import torch
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
import os
from MetaCYP.tool import load_model, load_args, get_scaler

if __name__ == "__main__":
    import argparse
    def set_predict_argument():
        parser = argparse.ArgumentParser(description="Predictor arguments")
        parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
        parser.add_argument('--cuda', action='store_true', help="Flag to use GPU")
        parser.add_argument('--predict_path', type=str, default='default_path.csv', help="Path to the input data for prediction")
        return parser.parse_args()
    _args = set_predict_argument()
else:
    _args = type("Args", (object,), {})() 
    current_dir = os.path.dirname(__file__)
    _args.model_path = os.path.join(current_dir, "saved_models", "cyp2c19.pt")
    _args.cuda = True
    _args.predict_path = "default_path.csv" 

_scaler = get_scaler(_args.model_path)
_train_args = load_args(_args.model_path)
for key, value in vars(_train_args).items():
    if not hasattr(_args, key):
        setattr(_args, key, value)

_model = load_model(_args.model_path, _args.cuda) 

def _compute_hiv_reward_internal(smiles: str, model, scaler) -> float:

    if smiles is None:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    model.eval()
    with torch.no_grad():
        pred = model([smiles])
    pred = pred.data.cpu().numpy()

    if scaler is not None:
        ave, std = scaler
        pred = pred * std + ave

    return float(pred[0])

def compute_hiv_reward(smiles: str) -> float:
    return _compute_hiv_reward_internal(smiles, _model, _scaler)

if __name__ == "__main__":
    test_smiles_list = [
        "CNCc1cn(c(c1)c2ccccc2F)[S](=O)(=O)c3cccnc3",
        "CC1C2=CC=CC=C2CCN1C3=NC(=NC(=C3C)C)NC4=CC=C(C=C4)F.Cl",
        "CC1=C(C)C(N2C(C)C3=C(C=CC=C3)CC2)=NC(NC4=CC=C(F)C=C4)=N1",
        "Cc1c(nc(nc1N2CCc3ccccc3C2C)Nc4ccc(cc4)F)C",
        "Cc1c(nc(n1N2CCc3ccccc3[C@H]2)Nc4ccc(cc4)F)C",
        "CCn1cc(C(=O)N2CCCC(n3ccc4ccccc43)C2)ccc1=O",
        "CCC(NC(=O)NC(C)C(=O)Nc1ccccc1)c1cccnc1",
        "COc1cc(CNc2cnccn2)ccc1S(N)(=O)=O"
    ]
    for smi in test_smiles_list:
        reward_value = compute_hiv_reward(smi)
        print(f"SMILES: {smi}, Reward: {reward_value:.4f}")