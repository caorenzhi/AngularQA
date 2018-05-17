# AngularQA
AngularQA is a single-model quality assessment tool to evaluate quality of predicted protein structures. It is built on a new representation that converts raw atom information into a series of carbon-alpha (Cα) atoms with side-chain information, defined by their dihedral angles and bond lengths to the prior residue. An LSTM network is used to predict the quality by treating each amino acid as a time-step and consider the final value returned by the LSTM cells.


Developed by Prof. Renzhi Cao at Pacific Lutheran University
