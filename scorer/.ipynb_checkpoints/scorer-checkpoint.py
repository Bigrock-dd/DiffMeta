# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
from __future__ import print_function
from rdkit import Chem
import rdkit.Chem.QED as QED
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



import math
import os.path as op
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import rdkit.Chem.Descriptors as Desc
from six import iteritems
#from scorer.vinadock import vinadock
_fscores = None
#!/usr/bin/env python
import os

def standardize_mols(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        print('standardize_mols error')
        return None
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

    
def normalize_logP(logp, min_val=-2.0, max_val=6.0):
    norm = (logp - min_val) / (max_val - min_val)
    return max(0, min(1, norm))    
    

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro
def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2  # <- 2 is the *radius* of the circular fingerprint
    )
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (0. - sizePenalty - stereoPenalty -
              spiroPenalty - bridgePenalty - macrocyclePenalty)

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def get_scores(objective, mols, standardize=True):
    if standardize:
        mols = [standardize_mols(mol) for mol in mols]
    mols_valid = [mol for mol in mols if mol is not None]
    
    scores = [get_score(objective, mol) for mol in mols_valid]
    scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
    
    return scores

class SpacialScore:
    def __init__(self, smiles, mol, verbose=False):
        self.smiles = smiles
        self.mol = mol
        self.verbose = verbose

        self.hyb_score = {}
        self.stereo_score = {}
        self.ring_score = {} 
        self.bond_score = {}

        self.chiral_idxs = self.find_stereo_atom_idxs()
        self.doublebonds_stereo = self.find_doublebonds_stereo()
        self.raw_score = self.calculate_spacial_score()
        num_atoms = Desc.HeavyAtomCount(self.mol)
        theoretical_max = num_atoms * 256.0
        
        self.score = self.normalize_score(self.raw_score, min_val=0, max_val=theoretical_max)
        
        self.per_atom_score = (self.raw_score / max(1, num_atoms)) / 256.0
        if self.verbose:
            self.display_scores()

    def display_scores(self):
        print("SMILES:", self.smiles)
        print("Atom Idx".ljust(10, " "), end="")
        print("Element".ljust(10, " "), end="")
        print("Hybrid".ljust(10, " "), end="")
        print("Stereo".ljust(10, " "), end="")
        print("Ring".ljust(10, " "), end="")
        print("Neighbs".ljust(10, " "))
        print("".ljust(60, "-"))
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            print(str(atom_idx).ljust(10, " "), end="")
            print(str(atom.GetSymbol()).ljust(10, " "), end="")
            print(str(self.hyb_score[atom_idx]).ljust(10, " "), end="")
            print(str(self.stereo_score[atom_idx]).ljust(10, " "), end="")
            print(str(self.ring_score[atom_idx]).ljust(10, " "), end="")
            print(str(self.bond_score[atom_idx]).ljust(10, " "))
        print("".ljust(60, "-"))
        print("Total Spacial Score:", self.score)
        print("Per-Atom Score:", round(self.per_atom_score, 2), "\n")

    def find_stereo_atom_idxs(self, includeUnassigned=True):
        stereo_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=includeUnassigned)
        return [atom_idx for atom_idx, _ in stereo_centers]

    def find_doublebonds_stereo(self):
        db_stereo = {}
        for bond in self.mol.GetBonds():
            if str(bond.GetBondType()) == "DOUBLE":
                db_stereo[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = bond.GetStereo()
        return db_stereo

    def calculate_spacial_score(self):
        score = 0
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            self.hyb_score[atom_idx] = self._account_for_hybridisation(atom)
            self.stereo_score[atom_idx] = self._account_for_stereo(atom_idx)
            self.ring_score[atom_idx] = self._account_for_ring(atom)
            self.bond_score[atom_idx] = self._account_for_neighbours(atom)
            score += self._calculate_score_for_atom(atom_idx)
        return score

    def _calculate_score_for_atom(self, atom_idx):
        return (self.hyb_score[atom_idx] *
                self.stereo_score[atom_idx] *
                self.ring_score[atom_idx] *
                self.bond_score[atom_idx])

    def _account_for_hybridisation(self, atom):
        hybridisations = {"SP": 1, "SP2": 2, "SP3": 3}
        hyb_type = str(atom.GetHybridization())
        return hybridisations.get(hyb_type, 4)

    def _account_for_stereo(self, atom_idx):
        if atom_idx in self.chiral_idxs:
            return 2
        for bond_atom_idxs, stereo in self.doublebonds_stereo.items():
            if atom_idx in bond_atom_idxs and not str(stereo).endswith("NONE"):
                return 2
        return 1

    def _account_for_ring(self, atom):
        if atom.GetIsAromatic():
            return 1
        return 2 if atom.IsInRing() else 1

    def _account_for_neighbours(self, atom):
        return (len(atom.GetNeighbors())) ** 2
    

    def normalize_score(self, score, min_val=0, max_val=600):
        norm_score = (score - min_val) / (max_val - min_val)
        return max(0, min(1, norm_score))    
    

def get_score(objective, mol):
    try:
        if objective == 'qed':
            return QED.qed(mol)
        elif objective == 'sa':
            x = calculateScore(mol)
            return (10. - x) / 9.   # normalized to [0, 1]
        elif objective == 'spacial':
            sps = SpacialScore(Chem.MolToSmiles(mol), mol)
            return sps.score * 10
        elif objective == 'logP':
            raw_logp = Desc.MolLogP(mol)
            return normalize_logP(raw_logp)
        else:
            raise NotImplementedError
    except (ValueError, ZeroDivisionError):
        return 0.

if __name__ == "__main__":
    test_smiles = "BrC(C(OCC1=CC=CC=C1)=O)/C(C(OC)OC)=C/C(OCC2=CC=CC=C2)=O"  
    test_mol = Chem.MolFromSmiles(test_smiles)
    if test_mol is None:
        print("Mol failure")
    else:
        spacial_value = get_score('spacial', test_mol)
        print("Spacial score for {}: {}".format(test_smiles, spacial_value))