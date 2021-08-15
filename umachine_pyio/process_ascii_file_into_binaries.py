"""
"""
import numpy as np
import gzip
from collections import OrderedDict
from .sf_history_header_processing import retrieve_dtype
from .memmap_array_utils import write_structured_array_to_memmap


def write_ascii_to_memmap_tree(
    sf_history_ascii_fname,
    sf_history_column_info_fname,
    output_dirname,
    requested_colnames,
    stellar_mass_cut,
):
    """"""
    colnums_to_yield, data_array_indices = _determine_colnums_to_yield(
        requested_colnames, sf_history_ascii_fname, sf_history_column_info_fname
    )

    full_columns_dict = _build_colnums_dict(sf_history_ascii_fname)
    stellar_mass_colnum = full_columns_dict["obs_sm"][0]

    raw_data_array = np.array(
        list(
            _stellar_mass_cut_data_generator(
                sf_history_ascii_fname,
                stellar_mass_colnum,
                stellar_mass_cut,
                colnums_to_yield,
            )
        )
    )

    assert len(raw_data_array) > 0, "Zero rows pass the M* cut"

    for colname in requested_colnames:
        ifirst, ilast = data_array_indices[colname]
        dt = np.dtype(retrieve_dtype(sf_history_column_info_fname, colname))

        if ifirst == ilast:
            data = np.array(raw_data_array[:, ifirst], dtype=dt)
        else:
            data = np.array(raw_data_array[:, ifirst : ilast + 1], dtype=dt)

        write_structured_array_to_memmap(data, output_dirname, colname)


def _stellar_mass_cut_data_generator(
    sf_history_ascii_fname, stellar_mass_colnum, stellar_mass_cut, colnums_to_yield
):
    """"""
    opener = _compression_safe_opener(sf_history_ascii_fname)
    with opener(sf_history_ascii_fname, "r") as fileobj:

        # First skip the header
        while True:
            try:
                raw_line = next(fileobj)
                if raw_line[0] != "#":
                    break
            except StopIteration:
                msg = "The {0} file contains only header information".format(
                    sf_history_ascii_fname
                )
                raise ValueError(msg)

        # Iterate over the data and yield the rows passing the M* cut
        line = (
            raw_line.strip().split()
        )  # First non-header line has already been retrieved
        while True:
            try:
                if float(line[stellar_mass_colnum]) >= stellar_mass_cut:
                    yield tuple(line[i] for i in colnums_to_yield)
                line = next(fileobj).strip().split()
            except StopIteration:
                break


def _colnums_to_yield_from_dict(columns_dict):
    colnums_to_yield = []
    for colname, colnums in columns_dict.items():
        ifirst, ilast = colnums
        colnums_to_yield.extend(range(ifirst, ilast + 1))
    return colnums_to_yield


def _data_array_indices_from_dict(columns_dict):
    data_array_indices = OrderedDict()
    icur = 0
    for colname in columns_dict.keys():

        colname_ifirst, colname_ilast = columns_dict[colname]
        data_array_indices[colname] = (icur, icur + (colname_ilast - colname_ifirst))
        icur += (colname_ilast - colname_ifirst) + 1
    return data_array_indices


def _determine_colnums_to_yield(colnames, fname, dtype_fname):
    """Return a list of integers storing the column numbers of the ASCII history file
    corresponding to the ordered sequence determined by the input ``colnames``.
    """
    colnames = np.atleast_1d(colnames)
    colnames = [s.lower() for s in colnames]

    full_columns_dict = _build_colnums_dict(fname)
    columns_dict = OrderedDict()
    for colname in colnames:
        umachine_colname = umachine_colname_from_user_colname(
            colname, fname, dtype_fname
        )
        columns_dict[colname] = full_columns_dict[umachine_colname]
    colnums_to_yield = _colnums_to_yield_from_dict(columns_dict)
    data_array_indices = _data_array_indices_from_dict(columns_dict)

    return colnums_to_yield, data_array_indices


def umachine_colname_from_user_colname(user_colname, umachine_fname, dtype_fname):
    full_columns_dict = _build_colnums_dict(umachine_fname)
    user_colnames = []
    opener = _compression_safe_opener(dtype_fname)
    with opener(dtype_fname, "r") as f:
        for i, raw_line in enumerate(f):
            user_colnames.append(raw_line.strip().strip("#").split()[0])
    msg = "Input user_colname = ``{0}`` not found in input ``{1}`` file".format(
        user_colname, dtype_fname
    )
    assert user_colname in user_colnames, msg
    return list(full_columns_dict.keys())[user_colnames.index(user_colname)]


def _retrieve_colname_list(fname):
    opener = _compression_safe_opener(fname)
    with opener(fname, "r") as f:
        header = next(f)
    header = header.strip("\n")
    header = header.strip("#")
    return [s.lower() for s in header.split(" ")]


def _get_header_char(fname):
    opener = _compression_safe_opener(fname)
    with opener(fname, "r") as f:
        first_line = next(f)
    return first_line[0]


def _retrieve_scale_list_header_line(fname, header_char=None):
    if header_char is None:
        header_char = _get_header_char(fname)

    opener = _compression_safe_opener(fname)
    with opener(fname, "r") as f:
        current_char = header_char
        while current_char == header_char:
            line = next(f)
            current_char = line[0]
            line = line.strip("\n")

            if ("scale" in line) & ("list" in line):
                return line
    raise ValueError("Unable to determine list of scales")


def _retrieve_scale_list(fname):
    scale_list_string = _retrieve_scale_list_header_line(fname)
    scale_list = scale_list_string.split(" ")

    for i, s in enumerate(scale_list):
        try:
            _ = float(s)
            break
        except ValueError:
            pass

    return [float(s) for s in scale_list[i:]]


def _determine_num_cols_from_colname(colname, num_scales):
    if "num_scales" in colname:
        return num_scales
    else:
        return 1


def _build_colnums_dict(fname):
    colname_list = _retrieve_colname_list(fname)
    scale_list = _retrieve_scale_list(fname)
    num_scales = len(scale_list)

    columns_dict = OrderedDict()
    ifirst, ilast = 0, 0
    for colname in colname_list:
        num_cols = _determine_num_cols_from_colname(colname, num_scales)
        if num_cols == 1:
            ilast = ifirst
        else:
            ilast = ifirst + num_scales - 1

        columns_dict[colname] = (ifirst, ilast)
        ifirst = ilast + 1

    return columns_dict


def _compression_safe_opener(fname):
    """Determine whether to use *open* or *gzip.open* to read
    the input file, depending on whether or not the file is compressed.
    """
    f = gzip.open(fname, "r")
    try:
        f.read(1)
        opener = gzip.open
    except IOError:
        opener = open
    finally:
        f.close()
    return opener
