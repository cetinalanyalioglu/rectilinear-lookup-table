This python package provides the ```RectilinearLookupTable``` class to perform n-dimensional interpolation over rectilinear grids, in a convenient fashion that relies on fancy indexing. The low-level interpolation is carried out by ```scipy```.

# Installation

Clone this repository, ```cd``` into the cloned directory and install via ```pip```,

```bash
pip install .
```

# Getting started

## **Terminology**
An n-dimensional rectilinear lookup table consists of $N$ control variables (coordinate variables), which forms the *axes* of the lookup table, and $M$ dependent variables that each correspond to some scalar which is a function of these control variables. We will refer to the control variables as *axes* of the table.

Coordinates (span) of *axis* $i$ can be expressed as a vector $\mathbf{x_i}$ of length $n_i$ (number of points). **The axis vectors $\mathbf{x}_i$ can have arbitrary spacing distribution, with only requirement that they consist of unique values and arranged in a monotonically increasing fashion.**

Described structure forms a so-called rectilinear grid. For each field, this corresponds to an array of shape ```(n1, n2, ..., nN)``` where each index corresponds to a control variable. Therefore, a table of $N$ control variables and $M$ dependent variables can be described by $N+M$ arrays of shape ```(n1, n2, ..., nN)```. 
 
## **Initialization**

The initalization requires user to specify data, either as a numpy structured array or as a dictionary, and axes names. 

If data is to be specified as a dictionary, keys of the dictionary correspond to user given names for each field (including control variables), and values of the dictionary contain the numpy arrays which contain the corresponding data. Shape of each array should be the same, and equal to ```(n1, n2, ..., nn)```.

If the data is to be specified as a numpy structured array, the array should have a shape of ```(n1, n2, ..., nn)``` and should contain the field names in ```array.dtype.names```.

In conjunction with above nomenclature, let us consider a basic three-dimensional example. We start by generating some arbitrary data in the suitable format,

```python
>>> import numpy as np
>>> import numpy.lib.recfunctions as rf

>>> from rectilinear_lookup_table import RectilinearLookupTable

>>> # Create axis vectors xi
>>> x1 = np.sort(np.random.random(100))
>>> x2 = np.linspace(0., 1., 11)
>>> x3 = np.linspace(-1., 1., 21)

# Create the grid
X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")

# Evaluate some arbitrary functions on the grid
phi1 = X1**2 + X2**2 + X3**2
phi2 = np.sin(X1) + np.cos(X2) + np.exp(X3)

# Option 1: Specify data as a dictionary
data_dict = {"x": X1, "y": X2, "z": X3, "f": phi1, "g": phi2}

# Option 2: Specify data as numpy structured array
data_arr = rf.unstructured_to_structured(np.stack((X1, X2, X3, phi1, phi2), axis=-1), names=["x", "y", "z", "f", "g"])
```

Notice that the spacing does not have to be constant, we have used random numbers to generate $\mathbf{x}_1$. Now let us create an instance of ```RectilinearLookupTable``` using this data,

```python
>>> table = RectilinearLookupTable(data=data_dict, axes=["x", "y", "z"])
```

During instance creation, the given input will be automatically controlled to check if it forms a rectilinear grid system or not, and an error will be raised if the check fails. Following checks are performed,

1. Is the spacing distribution for each axes same along their indices? (e.g, for each slice)
2. Do any of the axes contain non-unique values?
3. Are all axes arranged in a monotonically increasing fashion?

After initialization, you can always use ```print(table)``` to retrieve information regarding size, memory usage, and lower/upper bounds of fields,

```python
>>>print(table)

RectilinearLookupTable
======================

Dimensions: (100, 11, 21) x 5
Storage:    1 MB 
Axes:       x, y, z

-------------------------------------Min.------------Max.-----
(0) x                           1.40613777e-02  9.95657582e-01
(1) y                           0.00000000e+00  1.00000000e+00
(2) z                          -1.00000000e+00  1.00000000e+00
f                               1.97722344e-04  2.99133402e+00
g                               9.22242661e-01  4.55739867e+00
--------------------------------------------------------------
```

>**NOTE:** If the "Storage" fields constaints the mark ```(*)```, this means that the underlying data array is a memory reference to another ```numpy``` array, and the displayed memory information might be inaccurate.

## **Interpolation**

The interppolation functionality is embedded into the ```__getitem__``` method of ```RectilinearLookupTable```, which supports many combinations and can be used through fancy-indexing. In general, this corresponds to invoking ```table[idx1, idx2, ..., idxn]``` and behaviour depends on type of each ```idx```.

Depending on the given set of indices, it will either return a new instance of ```RectilinearLookupTable```, a numpy structured array, or a dictionary.

> **NOTE:** Besides some special cases, the length of indices should be equal to length of axes (control variables).

All valid indexing schemes are explained below (considering above example).

### **Point value based on interger indices**

This does not involve any sort of interpolation and basically returns values corresponding to given indices.

Index type: all ```int``` <br>
Return type: ```dict```

```python
>>> table[0,3,1]

{'x': 0.02202770083118477,
 'y': 0.30000000000000004,
 'z': -0.9,
 'f': 0.9004852196039083,
 'g': 1.3839320683618954}
```

### **Retrieve point value based on coordinate**

Interpolated value for a given set of coordinates along each axis.

Index type: all ```float``` <br>
Return type: ```dict```

```python
>>> table[0.1, 0., -0.25]

{'x': 0.10000000000000002,
 'y': 0.0,
 'z': -0.25,
 'f': 0.07502758987982232,
 'g': 1.8796064919136264}
```

### **Retrieve single field as a standalone array**

Non-axis fields can be retrieved as a standalone numpy array (not structured). This type of access directly returns the actual array where data is internally stored, and allows modification of array values.

Index type: single ```str``` <br>
Return type: ```np.ndarray```

```python
>>> # Retrieve array associated with field "f"
>>> f = table["f"]

>>> # Modify (inplace) field "f"
>>> table["f"][table["f"] < 0.01] = 0.
```

### **Reduce fields**

This type of indexing serves to conveniently reduce the table to a given set of fields. Axes can not be removed or changed.

Index type: ```List[str]``` <br>
Return type: ```RectilinearLookupTable```

```python
>>> table[["f"]]

RectilinearLookupTable
======================

Dimensions: (100, 11, 21) x 4
Storage:    1 MB (*)
Axes:       x, y, z

-------------------------------------Min.------------Max.-----
(0) x                           2.28279437e-02  9.81991600e-01
(1) y                           0.00000000e+00  1.00000000e+00
(2) z                          -1.00000000e+00  1.00000000e+00
f                               5.21115013e-04  2.96430750e+00
--------------------------------------------------------------
```

Notice that field ```"g"``` is removed in the returned instance.

### **Slicing based on index**

Perform slicing by fixing index of an axis. This means the dimension of the table is reduced by 1, and the axis where slice is performed is no longer an *axis*.

Index type: when any index is ```int``` (excluding the case where all indices are ```int```) <br>
Return type: ```RectilinearLookupTable```

```python
>>> table[0, :, :]

RectilinearLookupTable
======================

Dimensions: (11, 21) x 5
Storage:    0 MB (*)
Axes:       y, z

-------------------------------------Min.------------Max.-----
x                               2.28279437e-02  2.28279437e-02
(0) y                           0.00000000e+00  1.00000000e+00
(1) z                          -1.00000000e+00  1.00000000e+00
f                               5.21115013e-04  2.00052112e+00
g                               9.31007708e-01  3.74110779e+00
--------------------------------------------------------------
```

Notice that, the dimensions and the axes of the returned instance are changed.

### **Slicing based on value**

Perform slicing by specifying a value for a given axis. This is very similar to above operation, the dimension of the table is reduced by 1, but this time the slice is generated via interpolation. Depending on the size of table, this operation can be quite expensive, as the required amount of interpolation operations is equal to number of points in the table divided by the length of axis that is being sliced.

Index type: when any index is ```float``` (excluding the case where all indices are ```float```) <br>
Return type: ```RectilinearLookupTable```

```python
>>> table[0.1, :, :]

RectilinearLookupTable
======================

Dimensions: (11, 21) x 5
Storage:    0 MB (*)
Axes:       y, z

-------------------------------------Min.------------Max.-----
x                               1.00000000e-01  1.00000000e-01
(0) y                           0.00000000e+00  1.00000000e+00
(1) z                          -1.00000000e+00  1.00000000e+00
f                               1.00466962e-02  2.01004670e+00
g                               1.00801291e+00  3.81811300e+00
--------------------------------------------------------------

# Note that below is not supported (usage of Ellipsis) and will raise a KeyError
>>> table[0.1, ...]
```

### **Resampling with integer slices**

Perform a resampling on given axes.

Index type: when any index is a ```slice``` with an integer value in at least one of its ```start```, ```stop``` and ```step``` attributes <br>
Return type: ```RectilinearLookupTable```

```python
# Below downsamples the first axis by a factor of 2
>>> table[::2, :, :]

RectilinearLookupTable
======================

Dimensions: (50, 11, 21) x 5
Storage:    0 MB (*)
Axes:       x, y, z

-------------------------------------Min.------------Max.-----
(0) x                           2.28279437e-02  9.81891382e-01
(1) y                           0.00000000e+00  1.00000000e+00
(2) z                          -1.00000000e+00  1.00000000e+00
f                               5.21115013e-04  2.96411069e+00
g                               9.31007708e-01  4.54983126e+00
--------------------------------------------------------------
```

### **Resampling (re-gridding) via list or array of coordinates**

Perform an interpolation based re-gridding operation using prescribed coordinates for given axes.

Index type: when any index is a ```list``` or ```np.ndarray``` <br>
Return type: ```RectilinearLookupTable```

```python
>>> table[[0.1, 0.2, 0.3, 0.4, 0.5], : ,:]

RectilinearLookupTable
======================

Dimensions: (5, 11, 21) x 5
Storage:    0 MB 
Axes:       x, y, z

-------------------------------------Min.------------Max.-----
(0) x                           1.00000000e-01  5.00000000e-01
(1) y                           0.00000000e+00  1.00000000e+00
(2) z                          -1.00000000e+00  1.00000000e+00
f                               1.00466962e-02  2.25006932e+00
g                               1.00801291e+00  4.19769074e+00
--------------------------------------------------------------
```

### **Combinations**

Almost all of the above index types can be combined. In fact, this is the principal idea behind ```RectilinearLookupTable```, to make such operations with a convenient syntax. Depending on the combination of given indices, the order of computation is arranged such that cheaper operations are carried out first.

```python
>>> # Fix first axis at a value of 0.5, perform a regridding on second axis, and slice the third axis from its last index
>>> table[0.5, [0.1, 0.2, 0.3] , -1]

RectilinearLookupTable
======================

Dimensions: (3,) x 5
Storage:    0 MB (*)
Axes:       y

-------------------------------------Min.------------Max.-----
x                               5.00000000e-01  5.00000000e-01
(0) y                           1.00000000e-01  3.00000000e-01
z                               1.00000000e+00  1.00000000e+00
f                               1.26006932e+00  1.34006932e+00
g                               4.15302723e+00  4.19269491e+00
--------------------------------------------------------------
```
