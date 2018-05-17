#!/bin/python3

import sys, os, re
from os import listdir
from os.path import isfile, join
import numpy as np
from math import pi as PI
from keras.models import load_model
import keras.backend as K
from collections import namedtuple
#from load_proteins import read_pos_file, read_ss_file, read_prx_files, pad_protein_data

from functools import partial

AA = namedtuple('AA', ['id', 'hphob', 'polar', 'charged'])
SEQ3 = {
    'ALA': AA( 1, True, False, False), 'CYS': AA( 2, False, True, False),
    'ASP': AA( 3, False, False, True), 'GLU': AA( 4, False, False, True),
    'PHE': AA( 5, True, False, False), 'GLY': AA( 6, True, False, False),
    'HIS': AA( 7, False, True, False), 'ILE': AA( 8, True, False, False),
    'LYS': AA( 9, False, False, True), 'LEU': AA(10, True, False, False),
    'MET': AA(11, True, False, False), 'ASN': AA(12, False, True, False),
    'PRO': AA(13, True, False, False), 'GLN': AA(14, False, True, False),
    'ARG': AA(15, False, False, True), 'SER': AA(16, False, True, False),
    'THR': AA(17, False, True, False), 'VAL': AA(18, True, False, False),
    'TRP': AA(19, False, True, False), 'TYR': AA(20, False, True, False)
}

SS = {'C': 1, 'H': 2, 'B': 3, 'E': 4, 'G': 5, 'I': 6, 'T': 7, 'S': 8}
# Regex to read the pos file, note that group 2, the amino acid, will always be read regardless of
#  the settings made, leaving a viable range of [3, 8] to choose from.
POS_FILE_REGEX = re.compile(r'^\s*(\d+),\s*([A-Z]{3}),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\n?')
POS_SUB_REGEX = re.compile(r'\.pos$')

RANGES = {
    4: (-PI, 2*PI),  # TAO 2
    5: (0.0, 2*PI),  # THETA
    7: (-PI, PI),    # PHI
    8: (0.0, PI)     # DELTA
}

def normalize_feature(val, fnum: int) -> float:
    """
    Normalize a feature value to be in the range [0, 1]. Note, if the value goes beyond the expected
    range, it will not be clamped.
    :param val: Value to be normalized
    :param fnum: Feature which was extracted (should be in range [3, 8])
    :return: Normalized value
    """
    assert 3 <= fnum <= 8
    if fnum not in RANGES: return float(val)
    min, max = RANGES[fnum]
    return (val - min) / (max - min)

def read_pos_file(path, input_features, use_aa_props=False, normalize_features=False, trim_ends=False):
    """
    Reads a position data file given its path
    :param path: Path to the pos file
    :param input_features: The features that should be read from each line, see POS_FILE_REGEX.
    :param use_aa_props: Whether the physical properties of the amino acid should be included.
    :param normalize_features: Set to true if values such as TAO or THETA should be normalized to the range [0, 1].
    :param trim_ends: Set to true if the first and last protein values should be removed.
    :return: Array in the shape (residues, features)
    """
    pos_data = []
    posfile = open(path)

    for line in posfile:
        match = re.fullmatch(POS_FILE_REGEX, line)
        if not match: continue

        aa = SEQ3[match.group(2)]
        residue_data = [aa.id]
        if use_aa_props:
            residue_data.extend([aa.hphob, aa.polar, aa.charged])

        for x in input_features:
            if x == 2: continue
            val = float(match.group(x))
            if normalize_features:
                val = normalize_feature(val, x)
            residue_data.append(val)

        pos_data.append(residue_data)
    posfile.close()
    if trim_ends and len(pos_data) > 2:
        pos_data = pos_data[1:-1]
    return pos_data

def read_ss_file(path, trim_ends=False, plen=None):
    """
    Reads a secondary structure data file given its path. Final array will have a secondary
    structure value for each of its residues.
    :param path: Path to the pos file
    :param trim_ends: Set to true if the first and last protein values should be removed.
    :param plen: Length of the corresponding position file (once trimmed if relevant), this is needed because sometimes there is an extra amino acid listed in the position data
    :return: Array in the shape (residues,)
    """
    ss_data = []
    ssfile = open(path)
    raw = ssfile.read().split('\n')[1]
    ssfile.close()
    for c in raw:
        ss_data.append(SS[c])

    if trim_ends and len(ss_data) > 2:
        # checked pos and mkdssp outputs, when length is one less, the pos file has an extra AA
        # not listed in the secondary structure output
        assert plen is not None
        if plen == len(ss_data) - 1:
            # don't trim the one at the end if we would just have to add one back
            ss_data = ss_data[1:]
        else:
            ss_data = ss_data[1:-1]
    elif plen is not None:
        if plen - 1 == len(ss_data):
            ss_data.append(0.0)
    return ss_data

def read_prx_files(path, prx_r, trim_ends=False):
    """
    Reads the proximity data files using numpy.
    :param path: Path to the generic prx file, radii will be added automatically
    :param prx_r: List of the radii values to load
    :param trim_ends: Set to true if the first and last protein values should be removed.
    :return: A list of numpy arrays, each the length of the protein
    """
    prxs = []
    for r in prx_r:
        prx = np.load(path + str(r).replace('.', '_'))
        prxs.append(prx[1:-1] if trim_ends else prx)
    return prxs

def pad_protein_data(proteins, protein_length, num_features):
    """
    Front-pads the protein data with zeros or cuts off the end values if it is too long and converts
    it to a numpy ndarray.
    :param proteins: Array of proteins in the shape (proteins, residues, num_features)
    :param protein_length: The three length all proteins should be
    :param num_features: Number of values stored for each residue (including the AA)
    :return: nothing, acts in-place
    """
    for i in range(len(proteins)):
        # Forward pad with 0s so most important data is at the end
        if len(proteins[i]) < protein_length:
            tmp = [[0 for _ in range(num_features)] for _ in range(len(proteins[i]), protein_length)]
            tmp.extend(proteins[i])
            proteins[i] = tmp
        elif len(proteins[i]) > protein_length:
            proteins[i] = proteins[i][:protein_length]

def pearson_correlation(y_true, y_pred):
    """
    Custom keras metric to calculate the correlation between the true and predicted values.
    :param y_true: Tensor of true values
    :param y_pred: Tensor of predicted values
    :return: Tensor of correlation between the true and predicted values
    """
    true_dif = y_true - K.mean(y_true)
    pred_dif = y_pred - K.mean(y_pred)

    numerator = K.sum(true_dif * pred_dif)

    true_denom = K.sqrt(K.sum(K.square(true_dif)))
    pred_denom = K.sqrt(K.sum(K.square(pred_dif)))
    denominator = true_denom * pred_denom

    pearsonr = numerator / denominator
    return pearsonr

def predict(protein, protein_length, num_input_features, ml_model):
    protein = [protein]
    pad_protein_data(protein, protein_length, num_input_features)

    batch = np.ndarray([1, protein_length, num_input_features], dtype=np.float32)
    for x, i in enumerate(protein[0]):
        for y, j in enumerate(i):
            batch[0][x][y] = j

    return ml_model.predict(batch)[0]

def main():

    if len(sys.argv) < 3:
        showExample()
        sys.exit(0)

    myInput = sys.argv[1]
    dirOut = sys.argv[2]

    if not os.path.exists(dirOut):
        os.system("mkdir "+dirOut)

    ModelsDir = dirOut+"/Models"
    if not os.path.exists(ModelsDir):
        os.system("mkdir "+ModelsDir)

    # now copy each file to ModelsDir
    if os.path.isdir(myInput):
        for f in listdir(myInput):
            if isfile(join(myInput, f)):
                os.system("cp "+join(myInput, f)+" "+ModelsDir)
    else:
        os.system("cp "+myInput+" "+ModelsDir)
    FeatureDIR = dirOut + "/FEATURES"
    if not os.path.exists(FeatureDIR):
        os.system("mkdir "+FeatureDIR)
    ### parameters setting #######
    SoftwareRoot = os.path.split(os.path.abspath(__file__))[0]  # just directory path
    DSSPEXE = SoftwareRoot +'/mkdssp'
    RUNDDSP = SoftwareRoot + '/Cao_run_dssp.py'
    PDB2POS = SoftwareRoot + '/pdb2pos'
    PROXPATH = SoftwareRoot + '/proxcalc.py'

    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # prevent using a GPU since it calls them one at a time

    params = dict()
    params['protein_length'] = 500
    params['input_features'] = {4, 5}
    params['normalize_features'] = True       # we will normatlize the feature values to [0,1]
    params['trim_ends'] = True                # we will delete the first and last angle information
    params['aa_props'] = True                 # we will use amino acid polar and other physichemical properties
    params['load_ss'] = True                  # we will use the secondary structure information parsed by dssp
    params['prx_r'] = [8.0, 12.0]             # we will use environment features with radii 8 and 12

    custom_objects = dict()
    custom_objects['pearson_correlation'] = pearson_correlation
    ml_model = load_model(SoftwareRoot+"/models/2017-10-22.h5", custom_objects=custom_objects)
    ### first generate the position data and prx data and then process the data
    predictions = dirOut + "/AngularPrediction.txt"      # the output prediction file
    with open(predictions, 'w', buffering=1) as predictions:
        for f in listdir(ModelsDir):
            if isfile(join(ModelsDir, f)):    # get the model
                #print("Now processing " + join(ModelsDir, f))
                modelName = f                    # model name
                PDBPath = join(ModelsDir, f)     # the pdb path
                OutPOS = FeatureDIR+'/'+modelName+'.pos' # output position file
                os.system(PDB2POS+' '+PDBPath+' '+OutPOS)
                OutSS = FeatureDIR+'/'+modelName+'.ss' # output ss file by DSSP
                os.system("python3 "+RUNDDSP+" "+DSSPEXE+" "+PDBPath+" "+OutSS)
                try:
                    protein = read_pos_file(OutPOS, params['input_features'],
                                     normalize_features=params['normalize_features'],
                                     use_aa_props=params['aa_props'],
                                     trim_ends=params['trim_ends'])
                except:
                    continue
                residue_n = len(protein)
                try:
                    ss_data = read_ss_file(OutSS, trim_ends=params['trim_ends'], plen=residue_n)
                except:
                    continue
                if residue_n != len(ss_data):
                    print("Warning, the length "+str(residue_n)+" of protein is not the same as secondary structure of "+str(len(ss_data))+" \nCheck:"+str(ss_data)+"\n"+"python3 "+RUNDDSP+" "+DSSPEXE+" "+PDBPath+" "+OutSS+"\n")
                    continue
                for i in range(residue_n):
                    protein[i].insert(1, ss_data[i])



                # now process prx file
                OutPRX = FeatureDIR+'/'+modelName+'.prx'  # output PRX file , we need 8 and 12
                os.system("python3 "+PROXPATH+" 8.0 "+PDBPath+" "+OutPRX)
                os.system("python3 "+PROXPATH+" 12.0 "+PDBPath+" "+OutPRX)
                try:
                    prxs = read_prx_files(OutPRX, params['prx_r'], trim_ends=params['trim_ends'])
                except:
                    continue
                for p in prxs:
                    if len(p) != residue_n:
                        print('{} has {} more than the proximity data'.format(model, residue_n - len(p)))
                        continue
                    elif p.shape != (residue_n,):  # verify there are no hidden dimensions
                        print(model + ' has an invalid prx file')
                        continue
                    for i in range(residue_n):
                        protein[i].append(p[i])
                num_input_features = len(params['input_features']) + len(params['prx_r']) + 1
                if params['aa_props']: num_input_features += 3
                if params['load_ss']: num_input_features += 1
                try:
                    predictions.write('{} {}\n'.format(
                         modelName,
                         predict(protein, params['protein_length'], num_input_features, ml_model)[0]
                    ))
                except:
                    continue

"""
This is a function to show users some examples to run this tool
"""
def showExample():
    print("This is AngularQA, it's going to make predictions for a single model or a folder with several models. You need a program called pdb2pos to process PDB model for position result, and a trained model called 2017-10-22.h5, an output folder")
    print("Dependency: Keras, python3 and h5py library")
    print("For example:")
    print("python3 "+sys.argv[0]+" ../test/T0759.pdb ../test/Prediction_singleModel")
    print("python3 "+sys.argv[0]+" ../test/Models ../test/Prediction_ModelPool")

if __name__ == "__main__":
    main()
