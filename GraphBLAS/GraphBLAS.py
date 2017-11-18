import importlib
import subprocess
import os, sys
import re
import numpy as np
from scipy import sparse

# dictionary of GraphBLAS modules
_gb = {}

# constants for Makefile
_types = {"BOOL": 1,
         "INT8": 2,
         "UINT8": 3,
         "INT16": 4,
         "UINT16": 5,
         "INT32": 6,
         "UINT32": 7,
         "INT64": 8,
         "INT": 8,
         "UINT64": 9,
         "FLOAT32": 10,
         "FLOAT64": 11}

_algs = {"NONE": 1,
        "BFS": 2,
        "SSSP": 3,
        "TRICOUNT": 4}

# get environment variables
_PYBIND = (
        subprocess.check_output(["python3", "-m", "pybind11", "--includes"])
        .decode("ascii").strip().split(" ")
    )

_PYEXT = (
        subprocess.check_output(["python3-config", "--extension-suffix"])
        .decode("ascii").strip()
    )

_CWD = os.getcwd()

def _get_module(dtype, alg="NONE"):
    m = "%s_%s" % (dtype, alg)

    # first look in dictionary
    try:
        module = _gb[m]
    except:
        # then check directory
        try:
            _gb[m] = importlib.import_module("lib.gb_%s_%s" % (_types[dtype], _algs[alg]))
            module = _gb[m]
        # finally build and return module
        except:
            _gb[m] = _build_module(dtype, alg)
            module = _gb[m]

    return module

def _build_module(dtype, alg, semiring):
    if not os.path.exists("lib"):
        os.makedirs("lib")

    FNULL = open(os.devnull, 'w')
    subprocess.call(["make", 
        "pybind", 
        "DTYPE=%s" % _types[dtype],
        "ALG=%s" % _algs[alg],
        "PYBIND1=%s" % _PYBIND[0], 
        "PYBIND2=%s" % _PYBIND[1], 
        "PYBIND3=%s" % _PYBIND[2], 
        "PYEXT=%s" % _PYEXT, 
        "DIR=%s/" % _CWD])#, stdout=FNULL)
        
    return importlib.import_module("lib.gb_%s_%s" % (_types[dtype], _algs[alg]))

type_match = re.compile(r"'(.+)'")
def _get_type(a):
    try:
        return (a.dtype, str(a.dtype).split('.')[-1].upper())
    except:
        return (type(a[0][0]), 
                (re.search(type_match, str(type(a[0][0])))
                .group(1).split('.')[-1].upper()))

class Matrix():
    def __init__(self, m):
        # get C++ module with declaration for Matrix class
        self.dtype, t = _get_type(m)
        print(t)
        a = _get_module(t)

        # construct from scipy sparse matrix
        if (sparse.issparse(m)):
            print("coordinate")
            d = m.tocoo(copy=False)
            self.mat = a.init_sparse_matrix(d.shape[0], d.shape[1], d.row, d.col, d.data)
        # construct from tuple of arrays
        else:
            self.mat = a.init_sparse_matrix(max(m[1]) + 1, max(m[2]) + 1, m[1], m[2], m[0])

    def __str__(self):
        return self.mat.__str__()

    def __mul__(self, o):
        return self.mat * o.mat

class Vector():
    def __init__(self, v):
        self.dtype, t = _get_type(v)
        a = _get_module(t)
        self.vec = a.Vector(t)

    def __str__(self):
        return self.vec.__str__()

def bfs_level(matrix, root):
    a = _get_module(matrix, "BFS")
    return a.bfs_level(matrix.mat, root.mat)

def sssp(matrix, paths):
    a = _get_module(matrix, "SSSP")
    return a.sssp(matrix.mat, paths.vec)

def triangle_count(l_matrix, u_matrix):
    a = _get_module(l_matrix, "TRICOUNT")
    return a.triangle_count_newGBTL(l_matrix.mat, u_matrix.mat)

