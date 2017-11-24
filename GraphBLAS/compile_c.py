import os, sys, subprocess, importlib, inspect
import numpy as np

# dictionary of GraphBLAS modules
_gb = {}

# mapping from python/numpy types to c types
_types = {bool: "bool",
          int: "int64_t",
          float: "double",
          np.bool: "bool",
          np.int8: "int8_t",
          np.uint8: "uint8_t",
          np.int16: "int16_t",
          np.uint16: "uint16_t",
          np.int32: "int32_t",
          np.uint32: "uint32_t",
          np.int64: "int64_t",
          np.uint64: "uint64_t",
          np.float32: "float",
          np.float64: "double"}

_algs = {None: 1,
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

# get module directory
_CWD = inspect.getfile(inspect.currentframe()).rsplit("/", 1)[0]
sys.path.append(_CWD)
_MODDIR = os.path.abspath(_CWD + "/lib")

_CWD = os.getcwd()

def _get_module(dtype, alg=None, semiring=(None,None,None)):
    module = "gb_%s" % _types[dtype]
    if alg is not None: module += "_" + alg
    if semiring != (None,None,None): module += "_" + "_".join(map(str,semiring))

    # first look in dictionary
    try:
        return _gb[module]
    except:
        # then check directory
        try:
            _gb[module] = importlib.import_module("GraphBLAS.lib.%s" % module)
            return _gb[module]
        # finally build and return module
        except:
            target = None
            if alg is not None: target = "algorithm"
            elif semiring != (None, None, None): target = "semiring"
            else: target = "container"
            _gb[module] = _build_module(target, module, dtype, alg, semiring)
            return _gb[module]

def _build_module(target, module, dtype, alg, semiring):
    if not os.path.exists(_MODDIR):
        os.makedirs(_MODDIR)

    FNULL = open(os.devnull, 'w')
    subprocess.call(["make", 
        target, 
        "MODULE=%s" % module,
        "DTYPE=%s" % _types[dtype],
        "ALG=%s" % alg,
        "ADD_BINARYOP=%s" % semiring[0],
        "ADD_IDENTITY=%s" % semiring[1],
        "MULT_BINARYOP=%s" % semiring[2],
        "MNAME=%s" % "_".join(map(str,semiring[:2])),
        "SRNAME=%s" % "_".join(map(str,semiring)),
        "PYBIND1=%s" % _PYBIND[0], 
        "PYBIND2=%s" % _PYBIND[1], 
        "PYBIND3=%s" % _PYBIND[2], 
        "PYEXT=%s" % _PYEXT, 
        "DIR=%s/" % _MODDIR], cwd=_MODDIR)#, stdout=FNULL)
        
    return importlib.import_module("lib.%s" % module)

def _get_type(a):
    # if a is a numpy/scipy array
    try:
        return a.dtype.type
    # if a is an N-D list/array
    except:
        while type(a) not in _types:
            a = a[0]
        return type(a)
