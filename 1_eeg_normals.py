'''
This code was adapted from Zadeh et al. (2025);

DOI: 10.1088/1741-2552/adab22

==============================================================================

Below is copied annotaion from Zadeh et al.:
    

Script provided by Axel Thielscher from Technical University of Denmark.

This script extracts from the Simbnibs charm's output file the location of EEG landmarks (EEG10), along with the normal vectors relative to the scalp, that later can be used for simulations in BabelBrain.

Run this with an virtual environment pointing to Simbnibs, for example:

conda activate /Users/moonjeong/Applications/SimNIBS-4.1/simnibs_env/
python /Users/moonjeong/Desktop/0_eeg_normals.py

Edit below paths to files.

==============================================================================

'''

import numpy as np
import argparse
from simnibs.mesh_tools import read_msh

SKIN = 1005
TISSUE = [SKIN]
TRIANGLES = 2
POSITION_RADIUS = 5


def read_eeg_positions(file_path):
    electrodes = {}
    with open(file_path) as f:
        for txt in f.readlines():
            electrode_type, *coords, electrode_name = txt.split(',')
            electrodes[electrode_name.replace('\n', '')] = list(map(float, coords))
    return electrodes


def normal_at_coordinate(mesh, coordinate, triangle_normals):
    bar = mesh.elements_baricenters()[:]
    max_dist = POSITION_RADIUS
    dist = np.linalg.norm(bar - coordinate, axis=1)
    elm = mesh.elm.elm_number[
        (dist <= max_dist) *
        np.isin(mesh.elm.tag1, TISSUE) *
        np.isin(mesh.elm.elm_type, TRIANGLES)
    ]
    return np.mean(triangle_normals[elm], axis=0)


def normals_at_eeg_coordinates(mesh, electrodes, triangle_normals):
    position_normals = []
    for target_position in list(electrodes.values()):
        region_normal = normal_at_coordinate(mesh, target_position, triangle_normals)
        position_normals.append(region_normal)
    return position_normals


def write_eeg_normals(output_file, electrodes, normals):
    with open(output_file, 'w') as f:
        f.write("Name,R,A,S,Nx,Ny,Nz\n")
        for (name, position), normal in zip(electrodes.items(), normals):
            position_str = ','.join(map(str, position))
            normal_str = ','.join(map(str, normal))
            f.write(f"{name},{position_str},{normal_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_mesh_file', nargs=1, help="CHARM's output msh file; for example, m2m_SDR_0p55/SDR_0p55.msh")
    parser.add_argument('eeg_positions_file', nargs=1, help="CHARM's outout EEG-10 positions file; for example, m2m_SDR_0p55/eeg_positions/EEG10-10_Neuroelectrics.csv")
    parser.add_argument('csv_output_file', nargs=1, help="output filename; for example, SDR_0p55_eeg_normals.csv")
    
    opt = parser.parse_args()
    
    
    subject_mesh_file = opt.subject_mesh_file[0]
    eeg_positions_file = opt.eeg_positions_file[0]
    csv_output_file = opt.csv_output_file[0]
    

    subject_mesh = read_msh(subject_mesh_file)
    electrodes = read_eeg_positions(eeg_positions_file)
    triangle_normals = subject_mesh.triangle_normals()
    position_normals = normals_at_eeg_coordinates(subject_mesh, electrodes, triangle_normals)
    position_normals = -np.array(position_normals) # point normals inward
    write_eeg_normals(csv_output_file, electrodes, position_normals)

