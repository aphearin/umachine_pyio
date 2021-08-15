"""
"""
import numpy as np


def colname_generator(column_info_fname):
    with open(column_info_fname, 'r') as f:
        for raw_line in f:
            yield raw_line.strip().strip('#').split()[0]


def retrieve_requested_colnames(requested_colnames, formatting_fname):
    """
    """
    available_colnames = list(colname_generator(formatting_fname))
    if requested_colnames == ["all"]:
        return available_colnames
    else:
        if type(requested_colnames) is not list:
            requested_colnames = list(requested_colnames)
        for colname in requested_colnames:
            msg = "Column ``{0}`` does not appear in the first column of ``{1}``".format(
                colname, formatting_fname)
            assert colname in available_colnames, msg
        return requested_colnames


def retrieve_column_numbers(column_info_fname, colname):
    """
    """
    with open(column_info_fname, 'r') as f:
        while True:
            try:
                raw_line = next(f)
                line = raw_line.strip().split()
                if line[0] == colname:
                    return int(line[1]), int(line[2])
            except StopIteration:
                raise ValueError("Column name {0} not found in {1}".format(
                    colname, column_info_fname))


def retrieve_dtype(column_info_fname, colname):
    with open(column_info_fname, 'r') as f:
        while True:
            try:
                raw_line = next(f)
                line = raw_line.strip().split()
                if line[0].lower() == colname.lower():
                    return np.dtype([(str(colname.lower()), line[1])])
            except StopIteration:
                raise ValueError("Column name {0} not found in {1}".format(
                    colname, column_info_fname))
