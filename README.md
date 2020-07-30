# AngularQA

AngularQA is a single-model quality assessment tool to evaluate quality of predicted protein structures. It is built on a new representation that converts raw atom information into a series of carbon-alpha (Cα) atoms with side-chain information, defined by their dihedral angles and bond lengths to the prior residue. An LSTM network is used to predict the quality by treating each amino acid as a time-step and consider the final value returned by the LSTM cells.

# Citation
--------------------------------------------------------------------------------------
Matthew Conover, Max Staples, Dong Si, Renzhi Cao. "AngularQA: Protein Model Quality Assessment with LSTM Networks", submitted, 2018.

# Test Environment
--------------------------------------------------------------------------------------
Ubuntu, Centos

# Requirements
--------------------------------------------------------------------------------------
(1). Python3.5

(2). TensorFlow 
```
sudo pip install tensorflow
```
GPU is NOT needed.

(3) Install Keras:
```
sudo pip install keras
```

(4) Install the h5py library:  
```
sudo pip install h5py
```

# Run software
--------------------------------------------------------------------------------------
You could provide one PDB format model or a folder with several PDB format models for this software. Here are examples to test:

#cd script

#python3 AngularQA.py ../test/T0759.pdb ../test/Prediction_singleModel

#python3 AngularQA.py ../test/Models ../test/Prediction_ModelPool

You should be able to find a file named AngularPrediction.txt in the output folder.


--------------------------------------------------------------------------------------
Developed by Matthew Conover and Prof. Renzhi Cao at Pacific Lutheran University:

Please contact Renzhi Cao for any questions: caora@plu.edu (PI)
