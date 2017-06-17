""" Module storing functions used to load the UniverseMachine mock into memory.
"""
import os
from time import time
import numpy as np
from astropy.table import Table
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
import fnmatch

from .memmap_array_utils import read_ndarray_from_memmap_sequence
from .directory_tree_utils import memmap_fname_iterator

default_galprops = list((
    'sfr_history_main_prog', 'sm_history_main_prog',
    'sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl', 'halo_id', 'upid',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'rvir', 'mvir', 'mpeak'))

__all__ = ('load_mock_from_binaries', 'value_added_mock')


def load_mock_from_binaries(subvolumes, root_dirname=None, galprops=default_galprops):
    """ Load the mock catalog into memory.

    Parameters
    ----------
    subvolumes : sequence of integers
        Sequence specifies which subvolumes will be used to load data,
        e.g., subvolumes=np.arange(144) to load all 144 subvolumes of
        ``behroozi17`` mock data into memory,
        or np.arange(20) to load only the first twenty.

    root_dirname : string, optional
        Name of the parent directory of the collection
        subdirectories with names ``subvol_0``, ``subvol_1``, ``subvol_2``, etc.
        Default value is set in `umachine_pyio.cfg`.

    galprops : sequence of strings, optional
        List of galaxy properties to include in the mock catalog.
        Each element of the ``galprops`` sequence must be the name of a
        subdirectory of each ``subvol_N`` where the Numpy binary of
        a galaxy property is stored.

    Returns
    -------
    mock : Astropy Table
        Table of mock galaxies with the requested properties from the requested subvolumes.
    """
    from . import conf
    if root_dirname is None:
        root_dirname = conf.umachine_mocks_root_dirname

    start = time()

    galprops = np.array(list(set(np.atleast_1d(galprops))))

    mock = Table()
    for galprop in galprops:
        fname_tuples = list(memmap_fname_iterator(root_dirname, galprop, *subvolumes))
        memmap_fnames = [t[0] for t in fname_tuples]
        shape_fnames = [t[1] for t in fname_tuples]
        arr = read_ndarray_from_memmap_sequence(memmap_fnames, shape_fnames)
        mock[galprop] = arr

    end = time()
    print("Total runtime = {0:.2f} seconds".format(end-start))
    return mock


def subdirname_generator(root_dirname, subdirname_filepat):
    """ Yield the absolute dirname of subdirectories of ``root_dirname`` matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for dirname in fnmatch.filter(dirlist, subdirname_filepat):
            yield os.path.join(root_dirname, dirname)


def list_available_columns(root_dirname=None):
    """
    """
    from . import conf
    if root_dirname is None:
        root_dirname = conf.umachine_mocks_root_dirname
    return list(str(os.path.basename(subdir)) for subdir in
        subdirname_generator(os.path.join(root_dirname, 'subvol_0'), '*'))


def get_snapshot_times(root_dirname=None):
    """ Determine the age of the Universe at each snapshot of the mock
    by searching in the standard location within the input ``root_dirname``
    for the Numpy binary data storing this information.

    Parameters
    ----------
    root_dirname : string
        Name of the parent directory of the collection
        subdirectories with names ``subvol_0``, ``subvol_1``, ``subvol_2``, etc.

    Returns
    -------
    cosmic_age : ndarray
        Array of shape (num_snapshots, ) of the age of the universe in units of Gyr
    """
    from . import conf
    if root_dirname is None:
        root_dirname = conf.umachine_mocks_root_dirname
    fname = os.path.join(root_dirname, 'simulation_data', 'snapshot_times.npy')
    assert os.path.isfile(fname), "snapshot_times.npy file does not exist in {0}".format(root_dirname)
    return np.load(fname)


def value_added_mock(mock, Lbox):
    """ From an input mock that has been loaded into memory by the
    `load_mock_from_binaries` function, add some convenience columns
    and apply periodic boundary conditions.

    Parameters
    ----------
    mock : Astropy Table
        Output of the `load_mock_from_binaries` function

    Lbox : float
        Size of the simulation box - used to apply periodic boundary conditions

    Returns
    -------
    value_added_mock : Astropy Table
        Value-added mock catalog that contains ``halo_hostid`` column;
        the ``rvir`` will be rescaled by 1000 to be in Mpc units;
        the ``host_halo_mvir`` and ``host_halo_rvir`` columns
        will be calculated and added.
    """

    xyz_keylist = ['x', 'y', 'z']
    mock_keylist = list(mock.keys())
    xyz_keys = [key for key in xyz_keylist if key in mock_keylist]
    for xyz_key in xyz_keys:
        vel_key = 'v' + xyz_key

        if vel_key in list(mock.keys()):
            mock[xyz_key], mock[vel_key] = enforce_periodicity_of_box(mock[xyz_key],
                        Lbox, velocity=mock[vel_key],
                    check_multiple_box_lengths=True)
        else:
            mock[xyz_key] = enforce_periodicity_of_box(mock[xyz_key], Lbox,
                check_multiple_box_lengths=True)

    mock['halo_hostid'] = mock['halo_id']
    satmask = mock['upid'] != -1
    mock['halo_hostid'][satmask] = mock['upid'][satmask]

    idxA, idxB = crossmatch(mock['halo_hostid'], mock['halo_id'])

    try:
        mock['rvir'] = mock['rvir']/1000.
        mock['host_halo_rvir'] = mock['rvir']
        mock['host_halo_rvir'][idxA] = mock['rvir'][idxB]
    except KeyError:
        pass

    try:
        mock['host_halo_mvir'] = mock['mvir']
        mock['host_halo_mvir'][idxA] = mock['mvir'][idxB]
    except KeyError:
        pass

    return mock
