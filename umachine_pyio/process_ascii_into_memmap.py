"""
"""
import numpy as np
from collections import OrderedDict
from .memmap_array_utils import write_structured_array_to_memmap


def write_ascii_to_memmap_tree(
    sfh_ascii_fname, column_info_fname, output_dirname, requested_colnames=None
):
    """Read SFH ASCII data output from umachine and write to memmap column store"""

    columns_dict = _build_colnums_dict(sfh_ascii_fname, column_info_fname)

    if requested_colnames is None:
        requested_colnames = list(columns_dict.keys())

    data_array_indices = _data_array_indices_from_dict(columns_dict, requested_colnames)
    colnums_to_yield = _determine_colnums_to_yield(columns_dict, requested_colnames)
    raw_data_array = np.array(
        list(_ascii_data_iterator(sfh_ascii_fname, colnums_to_yield))
    )

    for colname in requested_colnames:
        ifirst, ilast = data_array_indices[colname]
        dt = np.dtype([(colname, columns_dict[colname][0])])

        if ifirst == ilast:
            data = np.array(raw_data_array[:, ifirst], dtype=dt)
        else:
            a, b = ifirst, ilast + 1
            data = np.array(raw_data_array[:, a:b], dtype=dt)

        write_structured_array_to_memmap(data, output_dirname, colname)


def _determine_colnums_to_yield(columns_dict, requested_colnames):
    colnums_to_yield = []
    for colname in requested_colnames:
        ifirst, ilast = columns_dict[colname][1:]
        for i in range(ifirst, ilast + 1):
            colnums_to_yield.append(i)
    return colnums_to_yield


def _ascii_data_iterator(sf_history_ascii_fname, colnums_to_yield):
    """"""
    with open(sf_history_ascii_fname, "r") as fileobj:
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
                yield tuple(line[i] for i in colnums_to_yield)
                line = next(fileobj).strip().split()
            except StopIteration:
                break


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


def _get_header_char(fname):
    with open(fname, "r") as f:
        first_line = next(f)
    return first_line[0]


def _retrieve_scale_list_header_line(fname, header_char=None):
    if header_char is None:
        header_char = _get_header_char(fname)

    with open(fname, "r") as f:
        current_char = header_char
        while current_char == header_char:
            line = next(f)
            current_char = line[0]
            line = line.strip("\n")

            if ("scale" in line) & ("list" in line):
                return line
    raise ValueError("Unable to determine list of scales")


def _build_colnums_dict(sfh_ascii_fname, column_info_fname):
    column_info_dict = _get_column_info_dict(column_info_fname)
    scale_list = _retrieve_scale_list(sfh_ascii_fname)
    num_scales = len(scale_list)

    columns_dict = OrderedDict()
    ifirst, ilast = 0, 0
    for colname, colinfo in column_info_dict.items():
        dt, is_history = colinfo

        if is_history:
            ilast = ifirst + num_scales - 1
        else:
            ilast = ifirst

        columns_dict[colname] = (dt, ifirst, ilast)
        ifirst = ilast + 1

    return columns_dict


def _get_column_info_dict(column_info_fname):
    column_info_dict = OrderedDict()
    with open(column_info_fname, "r") as f:
        for raw_line in f:
            line = raw_line.strip().split()
            colname = line[0]
            dt = line[1]
            is_history = bool(int(line[2]))
            column_info_dict[colname] = (dt, is_history)
    return column_info_dict


def _data_array_indices_from_dict(columns_dict, requested_colnames):
    data_array_indices = OrderedDict()
    icur = 0
    for colname in requested_colnames:
        colname_dt, colname_ifirst, colname_ilast = columns_dict[colname]
        data_array_indices[colname] = (icur, icur + (colname_ilast - colname_ifirst))
        icur += (colname_ilast - colname_ifirst) + 1
    return data_array_indices
