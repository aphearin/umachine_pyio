"""
"""
import os
from ..directory_tree_utils import sf_history_ascii_fname_iterator
from ..directory_tree_utils import subvol_dirname_iterator
from ..directory_tree_utils import memmap_fname_iterator
from ..directory_tree_utils import _infer_subvol_number_from_subvol_triplet
from ..directory_tree_utils import _infer_subvol_triplet_from_subvol_number


z0_root_dirname = "/Users/aphearin/work/UniverseMachine/data/0126_binaries/a_1.002310"
z0_ascii_root_dirname = "/Users/aphearin/work/random/ARCHIVES/2017/January17/0129"


def test_sf_history_ascii_fname_iterator():
    subvol_labels = -1
    root_dirname = z0_ascii_root_dirname
    prefix = "sfh_catalog_"
    suffix = ".txt"
    result = list(
        sf_history_ascii_fname_iterator(subvol_labels, root_dirname, prefix, suffix)
    )
    assert (
        0,
        os.path.join(z0_ascii_root_dirname, "sfh_catalog_1.002310.0.txt"),
    ) in result
    assert (
        1,
        os.path.join(z0_ascii_root_dirname, "sfh_catalog_1.002310.1.txt"),
    ) in result
    assert (
        2,
        os.path.join(z0_ascii_root_dirname, "sfh_catalog_1.002310.2.txt"),
    ) in result
    assert (
        3,
        os.path.join(z0_ascii_root_dirname, "sfh_catalog_1.002310.3.txt"),
    ) in result


def test_subvol_dirname_iterator1():
    result = list(subvol_dirname_iterator(z0_root_dirname, 1, 2, 100))
    assert "subvol_1" in result[0]
    assert "subvol_2" in result[1]
    assert "subvol_100" in result[2]


def test_subvol_dirname_iterator2():
    result = list(subvol_dirname_iterator(z0_root_dirname, *range(144)))
    assert len(result) == 144


def test_subvol_dirname_iterator3():
    result = list(subvol_dirname_iterator(z0_root_dirname, *range(5)))
    assert len(result) == 5


def test_memmap_fname_iterator1():
    result = list(memmap_fname_iterator(z0_root_dirname, "vx", 1, 2, 4))
    assert len(result) == 3


def test_memmap_fname_iterator2():
    result = list(memmap_fname_iterator(z0_root_dirname, "vx", *range(5)))
    assert len(result) == 5


def test_memmap_fname_iterator3():
    for memmap_fname, shape_fname in memmap_fname_iterator(
        z0_root_dirname, "vx", 1, 2, 4
    ):
        assert os.path.isfile(memmap_fname)
        assert os.path.isfile(shape_fname)


def test_subvol_triplet_number_consistency():
    ndiv_y, ndiv_z = 10, 10

    for subvol_num in range(20_000):
        ijk = _infer_subvol_triplet_from_subvol_number(subvol_num, ndiv_y, ndiv_z)
        inferred_subvol_num = _infer_subvol_number_from_subvol_triplet(
            ijk, ndiv_y, ndiv_z
        )
        assert inferred_subvol_num == subvol_num
