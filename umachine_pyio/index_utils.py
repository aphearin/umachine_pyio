"""Utility functions related to Rockstar/UniverseMachine indexing conventions
"""
import numpy as np


def get_num_snaps_since_orphan_merge(halo_id):
    """Calculate the number of snapshots since the time a subhalo merged

    Parameters
    ----------
    halo_id : ndarray of shape (n, )

    Returns
    -------
    num_snaps : ndarray of shape (n, )
        Integer array storing num_snaps since the time of merging
        Equals zero if and only if the (sub)halo survives to z=0

    """
    num_snaps = np.floor(halo_id / 1e15).astype(int)
    return num_snaps


def calculate_last_surviving_id(halo_id):
    """Calculate the halo ID of the last surviving progenitor halo

    Parameters
    ----------
    halo_id : ndarray of shape (n, )

    Returns
    -------
    last_surviving_id : ndarray of shape (n, )
        Integer array storing the halo_id of the last surviving progenitor halo
        Equals the input `halo_id` if and only if the (sub)halo survives to z=0

    """
    num_snaps_since_merge = get_num_snaps_since_orphan_merge(halo_id)
    last_surviving_id = halo_id - num_snaps_since_merge * int(1e15)
    return last_surviving_id
