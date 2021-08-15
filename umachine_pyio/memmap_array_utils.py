""" Module storing functions used to create memory maps to Numpy ndarrays,
stored with standardized filenames and metadata directory tree structure
to facilitate automated parallel I/O.
"""
import os
import numpy as np


def write_ndarray_to_memmap(arr, output_fname):
    """Create a Numpy memmap of the input array, additionally storing the
    shape and dtype in an ASCII file with filename pattern ``*_shape_and_dtype``,
    facilitating calculation of binary offsets.
    The read_shape_and_dtype_from_ascii function provides
    the necessary behavior to retrieve the shape and dtype from the file.

    Parameters
    ----------
    arr : ndarray
        Numpy array

    output_fname : string
        Filename of the memmap binary, including absolute path.
    """
    mmp = np.memmap(output_fname, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mmp[:] = arr[:]
    del mmp

    dirname = os.path.dirname(output_fname)
    basename = os.path.basename(dirname) + "_shape_and_dtype.txt"
    shape_and_dtype_output_fname = os.path.join(dirname, basename)
    write_shape_and_dtype_to_ascii(shape_and_dtype_output_fname, arr.shape, arr.dtype)


def write_shape_and_dtype_to_ascii(output_fname, shape, dtype):
    """Function creates an ASCII file that serves as metadata about a
    memory-mapped ndarray in a way that is readable by the
    `read_shape_and_dtype_from_ascii` function.

    Parameters
    ----------
    output_fname : string
        Name of the output ASCII metadata file

    shape : tuple
        Tuple storing the shape of the memory-mapped ndarray

    dtype : obj
        Instance of a Numpy dtype object.
    """

    line1 = "shape " + " ".join(str(i) for i in shape) + "\n"
    line2 = "dtype " + _unique_numpy_dtype_string(dtype) + "\n"

    with open(output_fname, "w") as f:
        f.write(line1)
        f.write(line2)


def read_shape_and_dtype_from_ascii(metadata_fname):
    """Function reads the input ASCII data and returns the shape and dtype
    of the ndarray stored in the same directory. See Notes below for description
    of the assumed format of the ASCII data.

    Examples
    --------
    >>> shape, dtype = read_shape_and_dtype_from_ascii(metadata_fname)  # doctest: +SKIP
    >>> mmp = np.memmap(mmp_fname, dtype=dtype, mode='r', shape=shape)  # doctest: +SKIP

    Parameters
    ----------
    metadata_fname : string
        Name of the input ASCII metadata file

    Returns
    -------
    shape : tuple
        Tuple storing the shape of the memory-mapped ndarray

    dtype : object
        Instance of a Numpy dtype object.

    Notes
    -----
    The assumed file format of the ASCII data storing the shape information is
    two rows and N columns, where N is the dimension of the Numpy array.
    Data in each row can be either space-, tab-, or comma-separated.

    The first row stores the shape of the array,
    For example, if the array stored some single property of 2315 galaxies,
    then the first row of ASCII would be ``shape 2315``.
    Or if the array stored 178 values for each galaxy, such as when storing the
    star-formation history of every galaxy in the Bolshoi-Planck simulation,
    then the first row would be ``shape 2315 178``.

    The second row stores the Numpy data type. So your second row should look
    something like, ``dtype f4`` or ``dtype i8``.
    """
    with open(metadata_fname, "r") as f:
        line1 = next(f).strip().split()
        line2 = next(f).strip().split()
    shape = tuple(int(i) for i in line1[1:])
    dtype = np.dtype(line2[1])
    return shape, dtype


def write_structured_array_to_memmap(arr, parent_dirname, *columns_to_save):
    """Function saves a memory map of the desired columns of a structured array
    according to the standard directory tree layout.

    Parameters
    ----------
    arr : array
        Numpy structured array

    parent_dirname : string
        Root directory where the data will be stored.

        Typically this is of the form 'some/path/subvol_0_1_2'.

    columns_to_save : sequence of strings, optional
        List of column names that will be memory-mapped to disk.
        If no argument is passed, default behavior is to store all columns.

    Notes
    -----
    For each memory-mapped column, an ASCII file serving as metadata
    will be stored in the same directory.
    This ASCII file stores the shape and dtype of the memory-mapped ndarray,
    facilitating calculation of binary offsets.
    The read_shape_and_dtype_from_ascii function provides
    the necessary behavior to retrieve the shape and dtype from the file.
    """
    dt = arr.dtype

    if len(columns_to_save) == 0:
        columns_to_save = ["all"]

    if columns_to_save[0] == "all":
        columns_to_save = dt.names

    for colname in columns_to_save:
        msg = "Column name ``{0}`` does not appear in input array".format(colname)
        assert colname in dt.names, msg

        output_dirname = os.path.join(parent_dirname, colname)
        try:
            os.makedirs(output_dirname)
        except OSError:
            pass

        output_fname = os.path.join(output_dirname, colname + ".memmap")
        write_ndarray_to_memmap(arr[colname], output_fname)


def determine_composite_shape_from_ascii_sequence(*shapes):
    """From an input sequence of shapes of Numpy arrays,
    determine the shape of the concatenated array, where concatenation is along
    axis-0.

    For example, for the input sequence of shapes
    ((100, 3), (110, 3), (105, 3)), the output tuple would be (355, 3).

    As another example, for the input sequence of shapes
    ((100, 3, 5), (110, 3, 5), (105, 3, 5)),
    the output tuple would be (355, 3, 5).

    So the first tuple-element of each shape can be any positive integer,
    but each other tuple-elements must be consistent with the corresponding
    tuple-element of all shapes in the input sequence.
    """
    msg = "Shapes must be compatible - only the first elements may differ"
    for shape in shapes:
        assert np.all(shape[1:] == shapes[0][1:]), msg

    first_dim = sum([shape[0] for shape in shapes])
    output_shape = [dim for dim in shapes[0][1:]]
    output_shape.insert(0, first_dim)
    return tuple(output_shape)


def read_ndarray_from_memmap_sequence(memmap_fnames, shape_fnames):
    """From an input sequence of filenames to memory-mapped Numpy arrays of known shape,
    return a single Numpy array storing the concatenation of these arrays.

    Parameters
    ----------
    memmap_fnames : sequence of strings
        Each string in the sequence will be treated as the filename of a
        Numpy binary storing a memory-mapped array

    shape_fnames : sequence of strings
        Each string in the sequence will be treated as the filename of
        ASCII data storing the shape and dtype of each Numpy array in the sequence.
        The ASCII data have a simple format described in the
        `read_shape_and_dtype_from_ascii` function documentation.

    Returns
    -------
    arr : ndarray
        Numpy array storing a concatenation of all the memory-mapped arrays
    """
    memmap_fnames = np.atleast_1d(memmap_fnames)
    shape_fnames = np.atleast_1d(shape_fnames)
    msg = "Must have the same number of ``shapes`` as ``memmap_fnames``"
    assert len(memmap_fnames) == len(shape_fnames), msg

    shapes = list(
        read_shape_and_dtype_from_ascii(shape_fname)[0] for shape_fname in shape_fnames
    )
    dt = read_shape_and_dtype_from_ascii(shape_fnames[0])[1]

    arr = np.empty(determine_composite_shape_from_ascii_sequence(*shapes), dtype=dt)

    ifirst = 0
    for fname, shape in zip(memmap_fnames, shapes):
        ilast = ifirst + shape[0]
        arr[ifirst:ilast] = np.memmap(fname, shape=shape, dtype=dt, mode="r")
        ifirst = ilast
    return arr


def _unique_numpy_dtype_string(dtype):
    """Private function providing a standardized string used to characterize
    a Numpy dtype
    """
    dt = np.dtype(dtype)
    try:
        s = dt[0].str
    except KeyError:
        s = dt.str
    return s[1:]
