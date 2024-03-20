import numpy as onp
import MDAnalysis
import glob
import os
from pathlib import Path
import csv

home = str(Path.home())


dataset_path = home + '/datasets/SPC_FW_All_AT_Reference_3nm_box'
# dataset_path = home + '/datasets/Butane_AT_Reference_Simulation_10ns'

os.chdir(dataset_path)
print(os.getcwd())

file_list_confs = glob.glob(dataset_path + '/Files/CG_files/*.gro')  # is not sorted
file_list_confs.sort()  # sorted configurations
file_list_forces = glob.glob(dataset_path + '/Files/CG_forces/*.dat')  # is not sorted
file_list_forces.sort()  # sorted forces

# get size of dataset:
n_snapshots = 10  # if we only want a subset; None will use all data
if n_snapshots == None:
    n_snapshots = len(file_list_confs)
universe = MDAnalysis.Universe(file_list_confs[0], file_list_confs[0])
n_atoms = universe.atoms.positions.shape[0]

# save configurations in python array
conf_data = onp.zeros([n_snapshots, n_atoms, 3])
force_data = onp.zeros([n_snapshots, n_atoms, 3])

for i, force_file_str in enumerate(file_list_forces[:n_snapshots]):
    forces = onp.loadtxt(force_file_str, delimiter=" ", skiprows=1)  # skips header
    forces = forces[forces[:, 0].argsort()]  # sort forces by CG index
    force_data[i, :, :] = forces[:, 1:]
    if i % 1000 == 0:
        print('Processed', i, 'force files!')

# fill array of configurations
for i, conf_file in enumerate(file_list_confs[:n_snapshots]):
    universe = MDAnalysis.Universe(conf_file, conf_file)
    conf_data[i, :, :] = universe.atoms.positions * 0.1  # convert to nm from MDAnalysis in A
    if i % 1000 == 0:
        print('Processed', i, 'conf files!')

onp.save(dataset_path + '/conf', conf_data, allow_pickle=False)  # save confs
onp.save(dataset_path + '/forces', force_data, allow_pickle=False)  # save confs

# load dataset via:
# confs = onp.load(dataset_path + '/conf.npy')
# forces = onp.load(dataset_path + '/forces.npy')
