"""
"""
import os
import fnmatch
from glob import glob


def sf_history_ascii_fname_iterator(subvol_labels, root_dirname, prefix, suffix):
    """ """
    if subvol_labels == -1:
        basename_filepat = prefix + "*" + suffix
        for path, dirlist, filelist in os.walk(root_dirname):
            for ascii_basename in fnmatch.filter(filelist, basename_filepat):
                ascii_fname = os.path.join(root_dirname, ascii_basename)
                subvol_index = int(ascii_basename.split(".")[-2])
                yield subvol_index, ascii_fname
    else:
        try:
            index_generator = (i for i in subvol_labels)
        except TypeError:
            index_generator = [subvol_labels]
        for subvol_index in index_generator:
            ascii_basename = prefix + "." + str(subvol_index) + suffix
            ascii_fname = os.path.join(root_dirname, ascii_basename)
            yield subvol_index, ascii_fname


def _get_subvol_id_strings(subvolumes):
    n_char = max([len(str(x)) for x in subvolumes])
    subvol_id_strings = [str(x).zfill(n_char) for x in subvolumes]
    return subvol_id_strings


def _infer_sorted_subvol_ids(drn):
    """Scan the drn for the list of all available subvolumes

    Parameters
    ----------
    drn : string
        Parent directory of all subdirectories such as 'subvol_123'

    Returns
    -------
    sorted_subvol_ids : list of strings
        If drn contains subvol_25, subvol_102, subvol_5 then
        function will return ['5', '25', ..., '102']
        Results will be sorted according to subvolume number

    """
    subdrn_list = glob(os.path.join(drn, "subvol_*"))
    subvol_ids = _get_subvol_id_strings([os.path.basename(s)[7:] for s in subdrn_list])
    subvol_ids = sorted(subvol_ids, key=lambda x: int(x))
    return subvol_ids


def subvol_dirname_iterator(root_dirname, *subvol_labels):
    """Generator yields a sequence of absolute paths where the data from
    each subvolume is stored

    Parameters
    ----------
    root_dirname : string
        Name of the parent directory of the collection
        subdirectories with names ``subvol_0``, ``subvol_1``, ``subvol_2``, etc.

    subvol_labels : sequence of integers
        Sequence defines which subvolume data will be loaded into memory

    """
    for subvol_label in subvol_labels:
        subvol_dirname = os.path.join(root_dirname, "subvol_" + str(subvol_label))

        msg = "\n{0} is not an existing directory\n".format(subvol_dirname)
        assert os.path.isdir(subvol_dirname), msg

        yield subvol_dirname


def memmap_fname_iterator(root_dirname, galprop_name, *subvol_labels):
    """Generator searches the input ``root_dirname`` for any subdirectory name
    matching the standard pattern, returning results only for the subvolumes
    with labels given by the input ``subvol_labels`` sequence.

    Parameters
    ----------
    root_dirname : string
        Name of the parent directory of the collection
        subdirectories with names ``subvol_0``, ``subvol_1``, ``subvol_2``, etc.

    galprop_name : string
        Name of the galaxy property (string must match the name of
        a subdirectory of each requested subvolume)

    subvol_labels : sequence of integers
        Sequence defines which subvolume data will be loaded into memory

    """
    for subvol_dirname in subvol_dirname_iterator(root_dirname, *subvol_labels):
        galprop_dirname = os.path.join(subvol_dirname, galprop_name)
        memmap_basename = galprop_name + ".memmap"
        memmap_fname = os.path.join(galprop_dirname, memmap_basename)
        shape_basename = galprop_name + "_shape_and_dtype.txt"
        shape_fname = os.path.join(galprop_dirname, shape_basename)
        yield memmap_fname, shape_fname
