from ase.io import read
import ase
import numpy as np
import nglview as nv
import networkx as nx


def get_bond_graph(atoms, bond_length=1.85):
    coords = atoms.get_positions()
    coords = np.broadcast_to(coords, (coords.shape[0], coords.shape[0], coords.shape[1]))
    distances = coords - np.transpose(coords, (1, 0, 2))
    del coords
    box_params = np.broadcast_to(atoms.get_cell().lengths(), distances.shape)
    distances = np.minimum(np.abs(distances), np.abs(box_params - np.abs(distances)))
    del box_params
    distances = np.sum(distances ** 2, axis=len(distances.shape) - 1) ** 0.5
    adj_matrix = distances
    del distances
    adj_matrix[adj_matrix > bond_length] = 0
    bond_graph = nx.from_numpy_matrix(adj_matrix)

    return bond_graph


def get_atom_idxs(atoms, atom_type):
    return np.nonzero(atoms.get_atomic_numbers() == atom_type)


def calc_carb_hybr(atoms):
    bond_graph = get_bond_graph(atoms)
    carbon_idxs = get_atom_idxs(atoms, atom_type=6)
    print(type(carbon_idxs))
    sp1 = []
    sp2 = []
    sp3 = []
    others = []
    for carbon in carbon_idxs:
        if len(list(bond_graph.neighbors(carbon))) == 2:
            sp1.append(carbon)
        elif len(list(bond_graph.neighbors(carbon))) == 3:
            sp2.append(carbon)
        elif len(list(bond_graph.neighbors(carbon))) == 4:
            sp3.append(carbon)
        else:
            others.append(carbon)

    sp1 = np.array(sp1)
    sp2 = np.array(sp2)
    sp3 = np.array(sp3)
    others = np.array(others)

    return (sp1, sp2, sp3, others)