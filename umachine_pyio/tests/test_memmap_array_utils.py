"""
"""
import numpy as np
from ..directory_tree_utils import memmap_fname_iterator
from ..memmap_array_utils import read_ndarray_from_memmap_sequence


z0_root_dirname = "/Users/aphearin/work/UniverseMachine/data/0126_binaries/a_1.002310"


def test_read_ndarray_from_memmap_sequence1():
    subvolumes = (1, 2, 3, 100)

    fname_tuples = list(memmap_fname_iterator(z0_root_dirname, "x", *subvolumes))

    memmap_fnames = [t[0] for t in fname_tuples]
    shape_fnames = [t[1] for t in fname_tuples]
    arr1 = read_ndarray_from_memmap_sequence(memmap_fnames, shape_fnames)
    assert np.all(arr1 >= 0)
    assert np.all(arr1 <= 250)

    fname_tuples = list(
        memmap_fname_iterator(z0_root_dirname, "vmax_at_mpeak_history", *subvolumes)
    )

    memmap_fnames = [t[0] for t in fname_tuples]
    shape_fnames = [t[1] for t in fname_tuples]
    arr2 = read_ndarray_from_memmap_sequence(memmap_fnames, shape_fnames)
    assert np.shape(arr2)[1] == 178

    assert arr1.shape[0] == arr2.shape[0]
