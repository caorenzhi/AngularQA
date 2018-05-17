#!/bin/python3

import sys, re
from collections import namedtuple
import numpy as np


# Residue seq(int), aa(str), xyz(np.array(3, dtype=float32)))
Residue = namedtuple('Residue', ['seq', 'aa', 'xyz'])


def read_pdb(input):
    """
    Read a pdb file from the input object specified.
    :param input: A valid file object.
    :return: An array of Residues
    """
    CA_REGEX = re.compile(r'^ATOM {2}.{5} {2}CA .(.{3}) .(.{4}). {3}(.{8})(.{8})(.{8})')

    residues = []
    for line in input:
        match = re.match(CA_REGEX, line)
        if match is None:
            continue
        residues.append(
            Residue(
                int(match.group(2)),
                match.group(1),
                np.array([float(match.group(x)) for x in range(3, 6)], dtype=np.float32)
            )
        )
    return residues


def validate_residues(residues):
    """ 
    Verify the validity of the residue sequence, namely that it has a Cα listed for all values in
    the sequence and is not missing any residues.
    :param residues: List of Residues sorted by ascending sequence number.
    :return: True if the sequence is complete, else false.
    """
    valid = True
    last = residues[0].seq
    for r in residues[1:]:
        if r.seq != last + 1:
            if r.seq == last:
                print('Multiple Cα listed for residue {}.'.format(r.seq), file=sys.stderr)
            elif r.seq - (last + 1) > 1:
                print('Missing sequences {} to {}.'.format(last + 1, r.seq), file=sys.stderr)
            else:
                print('Missing sequence {}.'.format(last + 1), file=sys.stderr)
            valid = False
        last = r.seq
    return valid


def extract_coordinates(residues):
    """
    Extract the xyz components from a list of Residues into a numpy ndarray.
    :param residues: List of Residues sorted by ascending sequence number.
    :return: Numpy ndarray with the shape (n, 3) where n = len(residues)
    """
    coords = np.empty((len(residues), 3), dtype=np.float32)
    for x, r in enumerate(residues):
        coords[x] = r.xyz
    return coords


def calculate_distances(coords):
    """
    Calculate the distances between all residues.
    :param coords: Coordinates of all residues in the shape (n, 3) where n = len(residues)
    :return: Numpy ndarray with the shape (n, n) where n = len(residues)
    """
    n = coords.shape[0]

    dists = np.empty((n, n), dtype=np.float32)

    for i in range(n):
        # calculate euclidean distance for this point p = coords[i] by finding the difference
        # between the x, y, and z pairs, and then calculate the norm for each set of differences
        # resulting in an array of distances from p to all other residues
        dists[i] = np.linalg.norm(coords - coords[i], axis=1)

    return dists


def count_adjacent(dists, radius, normalize=False):
    """
    Calculate the number of residues within a given radius of all residues.
    :param dists: Distances between all points
    :param radius: Radius in angstroms (Å) to consider the residues adjacent
    :param normalize: True if the counts should be normalized to the range [0, 1]
    :return: Array of the number of residues within a given radius of each residue
    """
    assert dists.shape[0] == dists.shape[1]
    assert radius > 0.0

    # subtract one from the count to exclude itself
    counts = (dists < radius).sum(axis=1) - 1

    if normalize:
        counts -= np.min(counts)
        max = np.float32(np.max(counts))
        if max != 0: # rare case, but handle it if all have the same count
            counts = counts.astype(np.float32, copy=False) / max
        else:
            counts = np.zeros(counts.shape, dtype=np.float32) + 0.5

    return counts


def proxcalc(input, output, radii, normalize=True):
    with open(input, 'r') as i:
        residues = read_pdb(i)

    if len(residues) == 0:
        print("Could not load pdb file " + input, file=sys.stderr)
        exit(1)

    residues.sort(key=lambda x: x[0])

    coords = extract_coordinates(residues)
    dists = calculate_distances(coords)

    for r in radii:
        counts = count_adjacent(dists, r, normalize=normalize)
        with open(output + str(r).replace('.', '_'), 'wb') as o:
            np.save(o, counts, allow_pickle=False)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        proxcalc(
            sys.argv[2],
            sys.argv[3],
            [float(x) for x in sys.argv[1].split(',')]
        )
    else:
        print("Usage: proxcalc.py <r> <input> <output>\n"
              " <r>      Radius or Radii to use, if multiple specify in the form:\n"
              "             1,2.2,3,4.123,5\n"
              " <input>  PDB to assess.\n"
              " <output> File to write the proxcount output to, will append the radius to the end.")
