#!/usr/bin/env python
from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import numpy as np
import json
import torch
import networkx as nx
import time
import re
from rdkit.Chem import AllChem
from scorer.scorer import get_scores
from scorer.vinadock import vinadock
import warnings
import random
from .ml_predictor_wrapper import compute_hiv_reward 
warnings.filterwarnings("ignore")
from datetime import datetime
import os

def get_novelty_in_df(df, sr=1., train_fps=None):
    if train_fps is not None:
        train_fps = random.sample(train_fps, int(sr * len(train_fps)))
    
    if 'sim' not in df.keys():
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['mol']]
        max_sims = []
        for i in range(len(gen_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], train_fps)
            max_sims.append(max(sims))
        df['sim'] = max_sims


def print_molecule_details(smiles_list, scores_dict, output_file=None):
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f = open(output_file, "w")
        print_func = lambda x: f.write(f"{x}\n")
    else:
        print_func = print
    

    header = "Index\tSMILES"
    for metric in scores_dict.keys():
        header += f"\t{metric}"
    print_func(header)
    
    for i, smiles in enumerate(smiles_list): 
        if smiles is None:
            continue
        line = f"{i}\t{smiles}"
        for metric, values in scores_dict.items():
            if i < len(values):
                line += f"\t{values[i]:.4f}" if isinstance(values[i], float) else f"\t{values[i]}"
            else:
                line += "\tN/A"
        print_func(line)
    
    if output_file:
        f.close()
        print(f"Detailed molecular scores have been saved to: {output_file}")        

def gen_score_list(protein, smiles, train_fps=None, weight_list=None, verbose=False):
    df = pd.DataFrame()
    if len(smiles) == 0:
        return []
    num_mols = len(smiles)

    while '' in smiles:
        smiles.remove('')
    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    validity = len(df) / (num_mols + 1e-8)
    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df, 0.5, train_fps)
    

    df['ml_score'] = [compute_hiv_reward(s) for s in df['smiles']]
    df = df[df['ml_score'] != -1] 

    ml_score = np.array(df["ml_score"])
    
    
    df['dock_score'] = vinadock(protein, df['smiles'].tolist())
    df = df[df['dock_score'] != -1] 
    df['dock_score_norm'] = np.clip(df['dock_score'], 0, 20) / 20 
    
    dock_score = np.array(df["dock_score_norm"])
    
    novelscore = 1 - df["sim"]
    df['qed'] = get_scores('qed', df['mol'])
    qedscore = np.array(df["qed"])
    df['sa'] = get_scores('sa', df['mol'])
    sascore = np.array(df["sa"])
    
    df['spacial'] = get_scores('spacial', df['mol'])
    spacial_score = np.array(df["spacial"])
    
    df['logP'] = get_scores('logP', df['mol'])
    logpscore = np.array(df["logP"])
    
    if weight_list is None:
        score_list = (0.1 * qedscore + 
                      0.1 * sascore + 
                      0.3 * novelscore + 
                      0.4 * ml_score +
                      0.4 * dock_score +
                      0.1 * spacial_score +
                      0.1 * logpscore)
    else:
        score_list = (weight_list[0] * qedscore +
                      weight_list[1] * sascore +
                      weight_list[2] * novelscore +
                      weight_list[3] * (1 - ml_score) +
                      weight_list[4] * dock_score +
                      weight_list[5] * spacial_score +
                      weight_list[6] * logpscore)
    
    score_list_final = score_list.tolist()

    
    if verbose:
        detail_dict = {
            'QED': qedscore.tolist(),
            'SA': sascore.tolist(),
            'Novelty': novelscore.tolist(),
            'ML_score' : ml_score.tolist(),
            'DS_Score': dock_score.tolist(),
            'Spacial': spacial_score.tolist(),
            'logP': df['logP'].tolist(),
            'Total': score_list_final
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"molecule_details_{protein}_{timestamp}.tsv"
        print_molecule_details(df['smiles'].tolist(), detail_dict, output_file)
    
    return score_list_final

def gen_score_disc_list(protein, smiles, thres=0.9, train_fps=None, weight_list=None, verbose=False):
    df = pd.DataFrame()
    if len(smiles) == 0:
        return []
    num_mols = len(smiles)

    while '' in smiles:
        smiles.remove('')
    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    validity = len(df) / (num_mols + 1e-8)
    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df, 0.5, train_fps)
    
    
    
    df['ml_score'] = [compute_hiv_reward(s) for s in df['smiles']]
    df = df[df['ml_score'] != -1] 

    ml_score = np.array(df["ml_score"])
    
    
    df['dock_score'] = vinadock(protein, df['smiles'].tolist())
    df = df[df['dock_score'] != -1] 
    
    if protein == '7w4a': hit_thr = 9.0 
    
    df['dock_score_norm'] = np.clip(df['dock_score'], 0, 20) / 10  
    
    dock_score = ((np.array(df['dock_score']) > (thres * hit_thr)).astype("float") *
           np.array(df['dock_score']) / 10)   
    

    
    get_novelty_in_df(df, 0.5, train_fps)
    novelscore = 1 - df["sim"]
    novelscore = (novelscore >= 0.4).astype("float") * novelscore
    
    df['qed'] = get_scores('qed', df['mol'])
    qedscore = np.array(df["qed"])
    df['sa'] = get_scores('sa', df['mol'])
    sascore = np.array(df["sa"])
    df['spacial'] = get_scores('spacial', df['mol'])
    spacial_score = np.array(df["spacial"])
    
    df['logP'] = get_scores('logP', df['mol'])
    logpscore = np.array(df["logP"])
    
    if weight_list is None:
        score_list = (0.1 * qedscore + 
                      0.1 * sascore + 
                      0.3 * novelscore + 
                      0.4 * ml_score +
                      0.4 * dock_score +
                      0.1 * spacial_score +
                      0.1 * logpscore)
    else:
        score_list = (weight_list[0] * qedscore +
                      weight_list[1] * sascore +
                      weight_list[2] * novelscore +
                      weight_list[3] * (1 - ml_score) +
                      weight_list[4] * dock_score +
                      weight_list[5] * spacial_score +
                      weight_list[6] * logpscore)
    
    score_list_final = score_list.tolist()
    
    if verbose:
        detail_dict = {
            'QED': qedscore.tolist(),
            'SA': sascore.tolist(),
            'Novelty': novelscore.tolist(),
            'ML_score' : ml_score.tolist(),
            'DS_Score': dock_score.tolist(),
            'Spacial': spacial_score.tolist(),
            'logP': df['logP'].tolist(),
            'Total': score_list_final
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"molecule_details_{protein}_{timestamp}.tsv"
        print_molecule_details(df['smiles'].tolist(), detail_dict, output_file)
    
    return score_list_final

def evaluate(protein, smiles, mols=None, train_fps=None, verbose=False):
    df = pd.DataFrame()
    if len(smiles) == 0:
        return dict.fromkeys([
            'validity', 'uniqueness', 'novelty', 'top_ds', 'hit',
            'avgscore', 'avgml', 'avgdock', 'avgqed', 'avgsa', 'avgspa', 'avglogP'
        ], 0)

    smiles = [s for s in smiles if s]
    df = pd.DataFrame({'smiles': smiles})
    num_mols = len(smiles)
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles] if mols is None else mols

    validity = len(df) / num_mols
    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df, 1.0, train_fps)
    novelty = len(df[df['sim'] < 0.6]) / len(df)

    df = df.drop_duplicates(subset='smiles')
    df['ml_score'] = [compute_hiv_reward(s) for s in df['smiles']]
    df = df[df['ml_score'] != -1]

    df['dock_score'] = vinadock(protein, df['smiles'].tolist())
    df = df[df['dock_score'] != -1]
    df['dock_score_norm'] = np.clip(df['dock_score'], 0, 20) / 10

    df['qed'] = get_scores('qed', df['mol'])
    df['sa'] = get_scores('sa', df['mol'])
    df['spacial'] = get_scores('spacial', df['mol'])
    df['logP'] = get_scores('logP', df['mol'])

    avgscore = (df['ml_score'] * df['dock_score_norm'] * df['qed'] * df['sa'] * df['logP']).mean()
    avgml = df['ml_score'].mean()
    avgdock = df['dock_score_norm'].mean()
    avgqed = df["qed"].mean()
    avgsa = df["sa"].mean()
    avgspa = df["spacial"].mean()
    avglogP = df["logP"].mean()

    hit_thr_map = {
        '7w4a': 9.0,'cyp1a2': 0.5, 'cyp2c19': 0.5, 'cyp2c9': 0.5, 'cyp3a4': 0.5, 'cyp2d6': 0.5
    }
    hit_thr = hit_thr_map.get(protein, 0.5)

    df = df[(df['qed'] > 0.5) & (df['sa'] > (10 - 5) / 9) & (df['sim'] < 0.6)]
    df = df.sort_values(by='dock_score', ascending=False)
    num_top5 = max(1, int(num_mols * 0.05))
    top_ds = (df.iloc[:num_top5]['dock_score'].mean(), df.iloc[:num_top5]['dock_score'].std())
    hit = len(df[df['dock_score'] > hit_thr]) / (num_mols + 1e-6)

    if verbose:
        detail_dict = {
            'QED': df['qed'].tolist(),
            'SA': df['sa'].tolist(),
            'Spacial': df['spacial'].tolist(),
            'logP': df['logP'].tolist(),
            'Sim': df['sim'].tolist(),
            'Novelty': (1 - df['sim']).tolist(),
            'ML_score': df['ml_score'].tolist(),
            'Dock_score': df['dock_score'].tolist(),
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_details_{protein}_{timestamp}.tsv"
        print_molecule_details(df['smiles'].tolist(), detail_dict, output_file)

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'top_ds': top_ds,
        'hit': hit,
        'avgscore': avgscore,
        'avgml': avgml,
        'avgdock': avgdock,
        'avgqed': avgqed,
        'avgsa': avgsa,
        'avgspa': avgspa,
        'avglogP': avglogP
    }

def evaluatelist(protein, smiles, mols=None, train_fps=None):
    df = pd.DataFrame()
    num_mols = len(smiles)
    if num_mols == 0:
        return {'validity': 0, 'uniqueness': 0,
                'novelty': 0, 'top_ds': 0, 'hit': 0,
                "avgscore": 0, "avgds": 0, "avgqed": 0, "avgsa": 0, "avglogP": 0}
    while '' in smiles:
        idx = smiles.index('')
        del smiles[idx]
        if mols is not None:
            del mols[idx]

    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles] if mols is None else mols

    get_novelty_in_df(df, 1.0, train_fps)

    df['ml_score'] = [compute_hiv_reward(s) for s in df['smiles']]
    df = df[df['ml_score'] != -1]

    df['dock_score'] = vinadock(protein, df['smiles'].tolist())
    df = df[df['dock_score'] != -1]
    df['dock_score_norm'] = np.clip(df['dock_score'], 0, 20) / 10

    df['qed'] = get_scores('qed', df['mol'])
    df['sa'] = get_scores('sa', df['mol'])
    df['spacial'] = get_scores('spacial', df['mol'])
    df['logP'] = get_scores('logP', df['mol'])

    del df['mol']
    return df

def gen_score_disc_listmose(protein, smiles, thres=0.9, train_fps=None, weight_list=None, verbose=False):
    if len(smiles) == 0:
        return []

    smiles = [s for s in smiles if s]
    num_mols = len(smiles)

    ml_scores = [compute_hiv_reward(s) for s in smiles]
    dock_scores = vinadock(protein, smiles)

    neg_prop = [(ml_scores[i] == -1 or dock_scores[i] == -1) for i in range(len(smiles))]

    valid_indexes = [i for i in range(len(smiles)) if not neg_prop[i]]
    if len(valid_indexes) == 0:
        print(f"[Warning] No valid molecules for scoring (protein: {protein})")
        return [-1] * num_mols

    valid_smiles = [smiles[i] for i in valid_indexes]
    valid_ml = [ml_scores[i] for i in valid_indexes]
    valid_dock = [dock_scores[i] for i in valid_indexes]
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]

    df = pd.DataFrame({'smiles': valid_smiles, 'mol': valid_mols})
    df['ml_score'] = valid_ml
    df['dock_score'] = valid_dock

    # --- Score Transform ---
    ml_score = (np.array(df['ml_score']) > (thres * 0.5)).astype(float) * np.array(df['ml_score'])

    hit_thr_map = {'7w4a': 9.0}
    hit_thr = hit_thr_map.get(protein, 9.0)
    dock_score = (np.array(df['dock_score']) > (thres * hit_thr)).astype(float) * (np.array(df['dock_score']) / 10)

    # --- Novelty ---
    get_novelty_in_df(df, 0.5, train_fps)
    novelscore = 1 - df["sim"]
    novelscore = (novelscore >= 0.4).astype(float) * novelscore

    # --- Other scores ---
    df['qed'] = get_scores('qed', df['mol'])
    df['sa'] = get_scores('sa', df['mol'])
    df['spacial'] = get_scores('spacial', df['mol'])
    df['logP'] = get_scores('logP', df['mol'])

    qedscore = np.array(df["qed"])
    sascore = np.array(df["sa"])
    spacial_score = np.array(df["spacial"])
    logpscore = np.array(df["logP"])

    # --- Final Score ---
    if weight_list is None:
        score_list = (
            0.1 * qedscore + 
            0.1 * sascore + 
            0.3 * novelscore + 
            0.3 * ml_score + 
            0.3 * dock_score + 
            0.1 * spacial_score + 
            0.1 * logpscore
        )
    else:
        score_list = (
            weight_list[0] * qedscore +
            weight_list[1] * sascore +
            weight_list[2] * novelscore +
            weight_list[3] * (1 - ml_score) + 
            weight_list[4] * dock_score +
            weight_list[5] * spacial_score +
            weight_list[6] * logpscore
        )

    valid_score_list = score_list.tolist()
    score_list_final = [-1] * len(neg_prop)
    pos = 0
    for idx, is_invalid in enumerate(neg_prop):
        if is_invalid:
            continue
        score_list_final[idx] = valid_score_list[pos]
        pos += 1

    if verbose:
        detail_dict = {
            'QED': qedscore.tolist(),
            'SA': sascore.tolist(),
            'Novelty': novelscore.tolist(),
            'ML_score': ml_score.tolist(),
            'Dock_score': dock_score.tolist(),
            'Spacial': spacial_score.tolist(),
            'logP': df['logP'].tolist(),
            'Total': valid_score_list
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"molecule_details_{protein}_{timestamp}.tsv"
        print_molecule_details(df['smiles'].tolist(), detail_dict, output_file)

    return score_list_final, df['dock_score'].tolist(), df['ml_score'].tolist()

def evaluatemose(protein, smiles, mols=None, train_fps=None, verbose=False):
    df = pd.DataFrame()
    if len(smiles) == 0:
        return {
            'validity': 0, 'uniqueness': 0, 'novelty': 0, 'top_ds': 0, 'hit': 0,
            'avgscore': 0, 'avgml': 0, 'avgdock': 0, 'avgqed': 0, 'avgsa': 0, 'avgspa': 0, 'avglogP': 0
        }

    smiles = [s for s in smiles if s]
    num_mols = len(smiles)
    df['smiles'] = smiles

    if mols is None:
        df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        df['mol'] = mols

    validity = len(df) / num_mols
    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df, 1., train_fps)
    novelty = len(df[df['sim'] < 0.6]) / len(df)

    df = df.drop_duplicates(subset=['smiles'])

    df['ml_score'] = [compute_hiv_reward(s) for s in df["smiles"].tolist()]
    df = df[df['ml_score'] != -1]
    df['dock_score'] = vinadock(protein, df["smiles"].tolist())
    df = df[df['dock_score'] != -1]
    df['dock_score_norm'] = np.clip(df['dock_score'], 0, 20) / 10

    df['qed'] = get_scores('qed', df['mol'])
    df['sa'] = get_scores('sa', df['mol'])
    df['spacial'] = get_scores('spacial', df['mol'])
    df['logP'] = get_scores('logP', df['mol'])

    avgscore = (df['ml_score'] * df['dock_score_norm'] * df["qed"] * df["sa"] * df["logP"]).mean()
    avgml = df['ml_score'].mean()
    avgdock = df['dock_score_norm'].mean()
    avgqed = df["qed"].mean()
    avgsa = df["sa"].mean()
    avgspa = df["spacial"].mean()
    avglogP = df["logP"].mean()

    hit_thr_map = {
        '7w4a': 9.0,'cyp1a2': 0.5, 'cyp2c19': 0.5, 'cyp2c9': 0.5, 'cyp3a4': 0.5, 'cyp2d6': 0.5
    }
    hit_thr = hit_thr_map.get(protein, 0.5)

    df = df[(df['qed'] > 0.5) & (df['sa'] > (10 - 5) / 9) & (df['sim'] < 0.6)]
    df = df.sort_values(by='dock_score', ascending=False)
    num_top5 = max(1, int(len(df) * 0.05))
    top_ds = (df.iloc[:num_top5]['dock_score'].mean(), df.iloc[:num_top5]['dock_score'].std())
    hit = len(df[df['dock_score'] > hit_thr]) / (len(df) + 1e-6)

    if verbose:
        detail_dict = {
            'QED': df['qed'].tolist(),
            'SA': df['sa'].tolist(),
            'Spacial': df['spacial'].tolist(),
            'logP': df['logP'].tolist(),
            'Sim': df['sim'].tolist(),
            'Novelty': (1 - df['sim']).tolist(),
            'ML_score': df['ml_score'].tolist(),
            'Dock_score': df['dock_score'].tolist()
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_details_{protein}_{timestamp}.tsv"
        print_molecule_details(df['smiles'].tolist(), detail_dict, output_file)

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'top_ds': top_ds,
        'hit': hit,
        'avgscore': avgscore,
        'avgml': avgml,
        'avgdock': avgdock,
        'avgqed': avgqed,
        'avgsa': avgsa,
        'avgspa': avgspa,
        'avglogP': avglogP,
        'molecule_scores': df[['smiles', 'dock_score', 'ml_score', 'qed', 'sa', 'spacial', 'logP']].copy()
    }

def evaluate_baseline(df, csv_dir, protein):
    from moses.utils import get_mol
    
    num_mols = 3000

    drop_idx = []
    mols = []
    for i, smiles in enumerate(df['smiles']):
        mol = get_mol(smiles)
        if mol is None:
            drop_idx.append(i)
        else:
            mols.append(mol)
    df = df.drop(drop_idx)
    df['mol'] = mols
    print(f'Validity: {len(df) / num_mols}')
    
    df['smiles'] = [Chem.MolToSmiles(m) for m in df['mol']]   

    print(f'Uniqueness: {len(set(df["smiles"])) / len(df)}')
    get_novelty_in_df(df)
    print(f"Novelty (sim. < 0.4): {len(df[df['sim'] < 0.4]) / len(df)}")

    df = df.drop_duplicates(subset=['smiles'])

    if protein not in df.keys():
        df[protein] = get_scores(protein, df['mol'])

    if 'qed' not in df.keys():
        df['qed'] = get_scores('qed', df['mol'])

    if 'sa' not in df.keys():
        df['sa'] = get_scores('sa', df['mol'])
    
    if 'logP' not in df.keys():
        df['logP'] = get_scores('logP', df['mol'])
    
    del df['mol']
    # df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == '7w4a': hit_thr = 9.0 
    
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.4]
    df = df.sort_values(by=[protein], ascending=False)

    num_top5 = int(num_mols * 0.05)

    top_ds = (df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std())
    hit = len(df[df[protein] > hit_thr]) / num_mols
    
    print(f'Novel top 5% DS (QED > 0.5, SA < 5, sim. < 0.4): {top_ds[0]:.4f} ± {top_ds[1]:.4f}')
    print(f'Novel hit ratio (QED > 0.5, SA < 5, sim. < 0.4): {(hit * 100):.4f} %')
    if 'logP' in df.columns:
        print(f'Average logP: {df["logP"].mean():.4f}')

if __name__=="__main__":
    smiles = ["Cc1noc(CCCN2CCC(=O)N(c3ccccc3O)CC2)n1", "CC(=O)c1cc2ccc1c(=O)n2-c1ccc(C(=O)NC(C)C)s1"]
    print(gen_score_list('parp1', smiles))