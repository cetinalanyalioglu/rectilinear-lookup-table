"""
RectilinearLookupTable
======================
"""

import numpy as np

from scipy.interpolate import interpn, interp1d
from typing import List, Union, Tuple


class RectilinearLookupTable:
    """A class to conveniently perform interpolation on n-dimensional rectilinear grids."""

    def __init__(self, data: Union[dict, np.ndarray], axes: List[str], check_rectilinear=True, **kwargs):
        """A class to conveniently perform interpolation on n-dimensional rectilinear grids.

        Parameters
        ----------
        data : Union[dict, np.ndarray]
            Input data, either numpy structured array or dictionary with keys as field names and values as arrays
        axes : List[str]
            Name of variables that form the axis (e.g control variables)
        check_rectilinear : bool, optional
            Check if given axes form a rectilinear grid, by default True
        """

        ##################
        # Validate input #
        ##################

        if not (isinstance(data, dict) or isinstance(data, np.ndarray)):
            raise TypeError("Argument data must be either of type dict or np.ndarray")

        if not isinstance(axes, list):
            raise TypeError("Argument axes must be a list of strings")

        if any([not isinstance(item, str) for item in axes]):
            raise TypeError("Argument axes must be a list of strings")

        # Convert dict input to structured array before further processing
        if isinstance(data, dict):
            data = RectilinearLookupTable.dict_to_structured(data)

        # Check if a name is specified for each axis
        if len(axes) != len(data.shape):
            raise ValueError("Given number of axes does not match with the shape of data")

        for item in axes:
            # Check if all axes are contained
            if item not in list(data.dtype.names):
                raise ValueError("Specified axis {:s} not found in given data".format(item))
            # Check if given input forms a rectilinear grid
            if check_rectilinear:
                if not RectilinearLookupTable.check_ax_rectilinear(data[item], axes.index(item), verbose=True):
                    raise ValueError("Given data and axes do not form a rectilinear grid")

        # We do not expose these to outside, as modification would break everything
        self.__data = data
        self.__axes = axes

        # Reserved
        self.__kwargs = kwargs

        # This vector stores the "coordinates" to avoid recomputation, this is an argument of scipy's interpn
        self.__coords = [self.get_axis(ax) for ax in self.__axes]

    def __getitem__(
        self, k: Union[int, float, str, list, tuple]
    ) -> Union[float, dict, np.ndarray, "RectilinearLookupTable"]:

        # The single integer index drops the first axis in numpy's implementation, but we will discard it
        if isinstance(k, int):
            raise KeyError("Single index of type int is not supported")

        # In our context, a float index without any axis specification does not mean anything
        elif isinstance(k, float):
            raise KeyError("Single index of type float is not supported")

        # Return the array corresponding to the given field, this is quite commonly used
        elif isinstance(k, str):
            if k in self.__axes:
                raise ValueError(
                    "This type of access to axis variables is not permitted, use RectilinearLookupTable.get_axes()"
                    " instead"
                )
            # Notice that we return the self.__data here, not self.data to allow user to modify non-axis fields
            return self.__data[k]

        # Single argument and it is a list, we only support list of strings that correspond to valid field names
        elif isinstance(k, list):
            # All elements of list are str, this corresponds to reduction of fields
            if all(isinstance(item, str) for item in k):
                # If for some reason, user attempts to request axes variables here, remove it to prevent duplication
                for ax in self.__axes:
                    try:
                        k.remove(ax)
                    except ValueError:
                        pass

                return RectilinearLookupTable(self.__data[self.__axes + k], self.__axes, check_rectilinear=False)

            else:

                raise KeyError(
                    "List based indexing can only be used to reduce table to given field names only, e.g"
                    ' RectilinearLookupTable[["field1", "field2"]] is valid'
                )

        # ========= #
        # Tuple key #
        # ========= #

        elif isinstance(k, tuple):
            # Check if size of axes is fine
            if len(k) != len(self.__axes):
                raise KeyError(
                    "The number of indices (or values) should be equal to number of axes ({:d})".format(
                        len(self.__axes)
                    )
                )
            # With the recursive strategy employed here, this is basically where we like to end up after several calls
            if all(isinstance(item, slice) for item in k):
                if (
                    all(isinstance(item.start, type(None)) for item in k)
                    and all(isinstance(item.step, type(None)) for item in k)
                    and all(isinstance(item.stop, type(None)) for item in k)
                ):
                    return self
            # All entries are integer, single value by index
            if all(isinstance(item, int) for item in k):
                return RectilinearLookupTable.structured_to_dict(self.__data[k])
                # return self.__data[k]
            # All entries are float, we are interpolating for a single coordinate
            if all(isinstance(item, float) for item in k):
                return self.interpolate(k)

            # =====================================================================================
            # For all other cases, we inspect contents of the tuple, and use the recursive strategy
            # =====================================================================================

            # Assign costs to compute cheaper operations first
            cost = np.zeros(len(k), dtype=int)

            for n, v in enumerate(k):
                # Cheapest operation is to slice the data from a given index of an axis
                if isinstance(v, int):
                    cost[n] = 1
                # Most expensive operation is to slice the data with interpolation
                elif isinstance(v, float):
                    cost[n] = 4
                elif isinstance(v, slice):
                    cost[n] = 2
                    # Indefinite slices should not be processed here, assign them the highest cost
                    if all(item is None for item in [v.start, v.stop, v.step]):
                        cost[n] = 5
                elif isinstance(v, list) or isinstance(v, np.ndarray):
                    cost[n] = 3
                else:
                    raise KeyError("Unsupported index type {:s} at position {:d}".format(type(v), n))

            # This is the key that is going to be processed first
            n = np.argmin(cost)
            v = k[n]

            # Integer index, this corresponds to a reduction of table dimension (slicing)
            if isinstance(v, int):
                # Pre-assemble a list corresponding to array[:, :, ..., :]
                idx = list((slice(None, None, None),) * len(self.__data.shape))
                # And place the integer index at the corresponding position
                idx[n] = v
                # Remove currently processed element from the original tuple
                request = list(k)
                request.remove(v)
                # And return a new instance with dropped axis
                return RectilinearLookupTable(
                    self.__data[tuple(idx)],
                    [ax for ax in self.__axes if ax != self.__axes[n]],
                    check_rectilinear=False,
                )[tuple(request)]

            # Float index, this corresponds to a reduction of table dimension with interpolation (slicing)
            if isinstance(v, float):
                # Pre-assemble a list corresponding to array[:, :, ..., :]
                idx = list((slice(None, None, None),) * len(self.__data.shape))
                # Place the corresponding index to an arbitrary position, as the value of corresponding axis
                # variable is going to be same along all of its indices after the interpolation
                idx[n] = 0
                # We create a meshgrid as the query, this consists of 4 vectors assembled in a tuple
                query = tuple(np.meshgrid(*self.__coords, indexing="ij"))
                # Then fix the corresponding coordinate to given float value at all points
                query[n][:] = v
                # Do the interpolation
                result = self.interpolate(query)
                # Remove currently processed element from original tuple based access
                request = list(k)
                request.remove(v)
                # Return a new instance with dropped axis
                return RectilinearLookupTable(
                    result[tuple(idx)], [ax for ax in self.__axes if ax != self.__axes[n]], check_rectilinear=False
                )[tuple(request)]

            # An (integer) slice - this corresponds to resampling over the selected axis
            if isinstance(v, slice):
                # Indefinite slices are skipped until all indices are indefinite slices
                if any(isinstance(item, int) for item in [v.start, v.step, v.stop]):
                    # Pre-assemble a list corresponding to array[:,:,...,:]
                    idx = list((slice(None, None, None),) * len(self.__data.shape))
                    # And place the slice index at the corresponding position
                    idx[n] = v
                    # Replace currently processed element from original tuple based access with ":"
                    request = list(k)
                    request[n] = slice(None, None, None)
                    # Return a new instance
                    return RectilinearLookupTable(self.__data[tuple(idx)], self.__axes, check_rectilinear=False)[
                        tuple(request)
                    ]

            # Convert lists to numpy arrays
            if isinstance(v, list):

                request = list(k)
                request[request.index(v)] = np.array(v)

                return RectilinearLookupTable(self.__data, self.__axes, check_rectilinear=False)[tuple(request)]

            # Multiple float values, this corresponds to re-gridding along the specified axis
            if isinstance(v, np.ndarray):
                # Get the original "coordinates" and modify the corresponding axis
                coords = list(self.__coords)
                coords[n] = v
                # We create a meshgrid as the query, this consists of 4 vectors assembled in a tuple
                query = tuple(np.meshgrid(*coords, indexing="ij"))
                # Do the interpolation
                result = self.interpolate(query)
                # Replace currently processed element from original tuple based access with ":"
                request = list(k)
                request[n] = slice(None, None, None)
                # Return a new instance with dropped axis
                return RectilinearLookupTable(result, self.__axes, check_rectilinear=False)[tuple(request)]

        else:
            raise KeyError("Unsupported index type {:s}".format(k))

    def __setitem__(self, k: str, v: np.ndarray):

        if not isinstance(k, str):
            raise KeyError(
                "This type of assignment only supports creation of new fields, e.g RectilinearLookupTable[new_field]"
                " = ..."
            )

        # Type and shape checking is performed in the routine
        self.__data = RectilinearLookupTable.structured_array_add_field(self.__data, k, v)

    def __delitem__(self, k):

        raise NotImplementedError("Item deletion is not supported yet")

    def __repr__(self):

        result = """
RectilinearLookupTable
======================

Dimensions: {:s} x {:d}
Storage:    {:.0f} MB {:s}
Axes:       {:s}

-------------------------------------Min.------------Max.-----""".format(
            str(self.shape),
            len(self.keys()),
            self.memsize,
            "(*)" if self.__data.base is not None else "",
            ", ".join(self.__axes),
        )

        for k in self.keys():
            result += """
{:30s} {:15.8e} {:15.8e}""".format(
                "({:d}) {:s}".format(self.__axes.index(k), k) if k in self.__axes else k,
                np.min(self.__data[k]),
                np.max(self.__data[k]),
            )

        result += """
{:s}""".format(
            62 * "-"
        )

        return result

    def __str__(self):

        return self.__repr__()

    @property
    def data(self) -> np.ndarray:
        """Returns a copy of numpy structured array that holds the data.

        Returns
        -------
        np.ndarray
            Data array
        """

        return np.copy(self.__data)

    @property
    def axes(self) -> List[str]:
        """Names of axes (e.g control variables) as a list of strings.

        Returns
        -------
        List[str]
            Axes names
        """

        return list(self.__axes)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of data array.

        Returns
        -------
        Tuple[int]
            Shape
        """

        return self.__data.shape

    @property
    def size(self) -> int:
        """Size of data array (e.g number of points).

        Returns
        -------
        int
            Size
        """

        return self.__data.size

    @property
    def memsize(self) -> float:
        """Size of data array in memory (MB).

        Returns
        -------
        float
            Memory size
        """

        return self.__data.nbytes / 1.0e6

    def get_axis(self, axis: Union[str, int]) -> np.ndarray:
        """Returns the selected axis as a vector, axis can either be specified by its name or its index.

        Parameters
        ----------
        axis : Union[str, int]
            Name or index of axis

        Returns
        -------
        np.ndarray
            Values of specified axis as a vector
        """

        if isinstance(axis, int):
            axis = self.__axes[axis]
        elif isinstance(axis, str):
            pass
        else:
            raise TypeError

        if axis not in self.__axes:
            raise ValueError

        return self.__get_axis_as_vector(axis)

    def interpolate(self, query: Union[Tuple, List, np.ndarray], **kwargs) -> np.ndarray:
        """Generalized interpolation routine that forms an interface to scipy.

        Parameters
        ----------
        query : Union[Tuple, List, np.ndarray]
            Coordinates to perform interpolation at. Length of query should be equal to number of axes.

        Returns
        -------
        np.ndarray
            Interpolated results for all fields assembled in a numpy structured array
        """

        # Check for garbage input
        if not (isinstance(query, tuple) or isinstance(query, list) or isinstance(query, np.ndarray)):
            raise TypeError(
                "Invalid type for query ({:s}). Supported types are tuple, list or np.ndarray".format(type(query))
            )

        # Make sure length of the query is equal to number of axes (e.g control variables)
        if len(query) != len(self.__axes):
            raise ValueError("Length of query should be equal to number of axis")

        # If all elements of the query are scalar, this is a single point interpolation, store results in a dict
        if all(np.isscalar(item) for item in query):
            result = dict.fromkeys(list(self.__data.dtype.names))

        # Interpolation using a given set of points, store results in a structured numpy array (native format)
        else:
            query = tuple([np.array(item) for item in query])

            # Make sure each element of this tuple has the same shape
            if not all(item.shape == query[0].shape for item in query):
                raise ValueError("Incosistent shape in elements of the query")

            # Now we can create a container array
            result = np.empty(query[0].shape, dtype=self.__data.dtype)

        # Now we can carry out the interpolation
        for name in self.__data.dtype.names:

            # For 1D interpolation, interpn raises an IndexError -- need to use a one-dimensional interpolator
            if len(self.__axes) == 1:
                interpolated = interp1d(self.get_axis(self.__axes[0]), self.__data[name], **kwargs)(np.array(query))
            else:
                interpolated = interpn(self.__coords, self.__data[name], query, **kwargs)

            # This is to avoid single-entry arrays in the output
            try:
                result[name] = interpolated.item()
            except ValueError:
                result[name] = interpolated

        return result

    def __get_axis_as_vector(self, axis: str) -> np.ndarray:
        """Internal routine to return specified axis as a vector."""

        # We will slice all other axes from their 0'th index
        idx = list((0,) * len(self.__data.shape))

        try:
            idx[self.__axes.index(axis)] = slice(None, None, None)
        except ValueError:
            raise ValueError("Given axis {:s} is not present in the list of fields".format(axis))

        return self.__data[axis][tuple(idx)]

    @staticmethod
    def check_ax_uniform(data: np.ndarray, axis_idx: int, verbose=False, **kwargs) -> bool:
        """Controls if spacing distribution of a given axis is constant across its dimension.

        Note
        ----
        The uniformity in this context does not mean spacing to be a constant value, it may be of varying "intervals",
        what this function does is to check if the "intervals" are the same across all slices or not.

        Parameters
        ----------
        data : np.ndarray
            Input array of n-dimensions, which contain data associated with given axis
        axis_idx : int
            The axis of the n-dimensional array to perform this check
        verbose : bool, optional
            Print information in case of false result, by default False

        Returns
        -------
        bool
            Uniform or not
        """

        idx = list((slice(None, None, None),) * len(data.shape))

        for n in range(data.shape[axis_idx]):

            idx[axis_idx] = n

            if not np.all(np.isclose(data[tuple(idx)].flatten()[0], data[tuple(idx)].flatten(), **kwargs)):

                if verbose:
                    print("Index {:d} at axis #{:d} is not rectilinear".format(n, axis_idx))

                return False

        return True

    @staticmethod
    def check_ax_unique(data: np.ndarray, axis_idx: int, verbose=False) -> bool:
        """Check if given axis contains only unique values.

        Note
        ----
        This check assumes that uniformity is already satisfied for given axis.

        Parameters
        ----------
        data : np.ndarray
            Input array of n-dimensions, which contain data associated with given axis
        axis_idx : int
            The axis of the n-dimensional array to perform this check
        verbose : bool, optional
            Print information in case of false result, by default False

        Returns
        -------
        bool
            Unique or not
        """

        idx = list((0,) * len(data.shape))

        idx[axis_idx] = slice(None, None, None)

        # In the following, "u" contains unique values and "c" contains respective counts of them
        u, c = np.unique(data[tuple(idx)], return_counts=True)

        if len(u) != data.shape[axis_idx]:

            if verbose:
                print("Following values axis #{:d} are duplicated:".format(axis_idx))
                for n in range(c.size):
                    if c[n] > 1:
                        print("* {:g} ({:d} times)".format(u[n], c[n]))

            return False

        return True

    @staticmethod
    def check_ax_monotonic(data: np.ndarray, axis_idx: int, verbose=False) -> bool:
        """Check if given axis contains monotonically increasing values.

        Note
        ----
        This check assumes that uniformity and uniqueness are already satisfied for given axis.

        Parameters
        ----------
        data : np.ndarray
            Input array of n-dimensions, which contain data associated with given axis
        axis_idx : int
            The axis of the n-dimensional array to perform this check
        verbose : bool, optional
            Print information in case of false result, by default False

        Returns
        -------
        bool
            Monotonic or not
        """

        idx = list((0,) * len(data.shape))

        idx[axis_idx] = slice(None, None, None)

        if np.any(np.diff(data[tuple(idx)]) <= 0.0):

            if verbose:
                print("Values for axis #{:d} is not monotonic".format(axis_idx))

            return False

        return True

    @staticmethod
    def check_ax_regular(data: np.ndarray, axis_idx: int, **kwargs) -> bool:
        """Check if given axis has constant spacing.

        Note
        ----
        This check assumes that uniformity is already satisfied for given axis.

        Parameters
        ----------
        data : np.ndarray
            Input array of n-dimensions, which contain data associated with given axis
        axis_idx : int
            The axis of the n-dimensional array to perform this check

        Returns
        -------
        bool
            Regular spacing or not
        """

        idx = list((0,) * len(data.shape))

        idx[axis_idx] = slice(None, None, None)

        if np.allclose(np.diff(data[tuple(idx)], n=2), np.zeros(data[tuple(idx)].size - 2), **kwargs):

            return True

        return False

    @staticmethod
    def check_ax_rectilinear(data: np.ndarray, axis_idx: int, verbose=False, **kwargs) -> bool:
        """Check if given axis is rectilinear.

        Parameters
        ----------
        data : np.ndarray
            Input array of n-dimensions, which contain data associated with given axis
        axis_idx : int
            The axis of the n-dimensional array to perform this check
        verbose : bool, optional
            Print information in case of false result, by default False

        Returns
        -------
        bool
            Rectilinear or not
        """

        return (
            RectilinearLookupTable.check_ax_uniform(data, axis_idx, verbose, **kwargs)
            and RectilinearLookupTable.check_ax_unique(data, axis_idx, verbose)
            and RectilinearLookupTable.check_ax_monotonic(data, axis_idx, verbose)
        )

    @staticmethod
    def structured_array_add_field(base: np.ndarray, name: str, array: np.ndarray) -> np.ndarray:
        """Add a new field to an existing numpy structured array.

        Parameters
        ----------
        base : np.ndarray
            Structured array to extend
        name : str
            Name of new field
        array : np.ndarray
            Data array associated with new field, must be of shape ```base.shape```

        Returns
        -------
        np.ndarray
            Extended array
        """

        if not isinstance(array, np.ndarray):
            raise TypeError("Argument array must be of type np.ndarray")

        if array.shape != base.shape:
            raise ValueError("Shape of given array does not match with base array")

        # Create a new empty array that contains the new field
        result = np.empty(
            array.shape,
            dtype={
                "names": [item[0] for item in base.dtype.descr] + [name],
                "formats": [item[1] for item in base.dtype.descr] + [array.dtype],
            },
        )

        # Assemble
        result[name] = array

        for k in base.dtype.names:
            result[k] = base[k]

        return result

    def keys(self) -> List[str]:
        """Returns field names present in the data as a list of strings.

        Returns
        -------
        List[str]
            Field names
        """

        return list(self.__data.dtype.names)

    def save_h5(self, filename: str) -> None:
        """Save the data and axes information of this instance to an HDF5 file, that can later be loaded using the
        ```RectilinearLookupTable.read_h5``` method.

        Parameters
        ----------
        filename : str
            Path to HDF5 file to be generated
        """

        try:
            import h5py
        except ImportError:
            raise ImportError("This feature requires h5py package")

        with h5py.File(filename, "a") as h5:

            # Save axes as a space-seperated string
            h5.attrs["axes"] = " ".join(self.__axes)

            # Save each field in a separate array
            for k in self.keys():
                h5.create_dataset(k, data=self.__data[k])

    @staticmethod
    def read_h5(filename: str, **kwargs) -> "RectilinearLookupTable":
        """Create an instance of ```RectilinearLookupTable``` by reading an HDF5 file pregenerated by the
        ```RectilinearLookupTable.save_h5``` method. Keyword arguments are passed to the initialization routine.

        Parameters
        ----------
        filename : str
            Path to HDF5 file

        Returns
        -------
        RectilinearLookupTable
        """

        try:
            import h5py
        except ImportError:
            raise ImportError("This feature requires h5py package")

        dtype = {k: [] for k in ["names", "formats"]}

        with h5py.File(filename, "r") as h5:

            axes = h5.attrs["axes"].split()

            # Read names and data dtypes
            for k in h5.keys():
                dtype["names"].append(k)
                dtype["formats"].append(h5[k].dtype)

            # Create a structured numpy array to store data and initialize RectilinearLookupTable
            data = np.empty(h5[axes[0]].shape, dtype=np.dtype(dtype))

            # Read each field
            for k in h5.keys():
                data[k] = h5[k][:]

        return RectilinearLookupTable(data=data, axes=axes, **kwargs)

    @staticmethod
    def dict_to_structured(data: dict) -> np.ndarray:
        """Converts arrays of uniform shape stored in a dictionary to a numpy structured array.

        Parameters
        ----------
        data : dict
            Dictionary with keys corresponding to field names and values corresponding to arrays

        Returns
        -------
        np.ndarray
            Corresponding numpy structured array
        """

        for k, v in data.items():

            if not isinstance(k, str):
                raise TypeError("Keys must be of type str, corresponding to names of fields")

            if not isinstance(v, np.ndarray):
                raise TypeError("Values must be of type np.ndarray")

            if v.shape != data[list(data.keys())[0]].shape:
                raise ValueError("Array shapes are not uniform")

        result = np.empty(
            data[list(data.keys())[0]].shape,
            dtype=np.dtype({"names": list(data.keys()), "formats": [item.dtype for item in data.values()]}),
        )

        for k, v in data.items():
            result[k] = v

        return result

    @staticmethod
    def structured_to_dict(data: np.ndarray) -> dict:
        """Converts numpy structured array to a dictionary.

        Parameters
        ----------
        data : np.ndarray
            Input array

        Returns
        -------
        dict
            Input array converted to a dict
        """

        try:
            if data.dtype.names is None:
                raise TypeError("Input must be a structured numpy array")
        except AttributeError:
            raise TypeError("Input must be a structured numpy array")

        result = dict.fromkeys(list(data.dtype.names))

        for k in result.keys():
            result[k] = data[k]

        return result
