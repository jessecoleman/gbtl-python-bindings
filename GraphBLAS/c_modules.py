import hashlib
import importlib
import inspect
import os
import subprocess
import sys
import zlib

# compiler flags
CXX         = "g++"
#CXX         = "clang"
LANG	    = "-std=c++14"
OPTS	    = "-O3 -march=native -DNDEBUG -shared -fPIC -fvisibility=hidden" #-flto=thin"
PICKY	    = "-Wall"
DEBUG 	    = "-g"
FLAGS       = "-DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL"

# project directories
CWD         = inspect.getfile(inspect.currentframe()).rsplit("/", 1)[0]
sys.path.append(CWD)

GB_SOURCE   = "/home/jessecoleman/gbtl/src/"
MODULES     = os.path.abspath("{cwd}/modules".format(cwd=CWD))
C_CODE      = os.path.abspath("{cwd}/c_code".format(cwd=CWD))
#TODO dynamically configure this with cmake
#PROJECT     = "-I{c_code}".format(c_code=C_CODE)
PROJECT     = "-I{gb_source} -I{c_code}".format(gb_source=GB_SOURCE, c_code=C_CODE)

# get environment variables
PYBIND      = (
        subprocess.check_output("python3 -m pybind11 --includes".split())
        .decode("ascii").strip()
)

# get file extension for modules
PYEXT       = (
        subprocess.check_output("python3-config --extension-suffix".split())
        .decode("ascii").strip()
)


def get_module(target, args, kwargs):

    module = zlib.adler32(
            "t{}a{}k{}".format(target, args, kwargs).encode("utf-8")
    )
#   module = hashlib.md5(str(args).encode("utf-8")).hexdigest()

    try:
        return importlib.import_module(
                "GraphBLAS.modules.{mod}".format(mod=module)
        )

    except (ImportError, ModuleNotFoundError):
        print("building module {}".format(target))

    if not os.path.exists(MODULES):
        os.makedirs(MODULES)

    cmd = [
            CXX,
            LANG,
            *OPTS.split(),
            *FLAGS.split(),
            *PYBIND.split(),
            #PICKY,
            #DEBUG,
            *PROJECT.split(),
            "-MT", "graphblas{pyext}".format(pyext=PYEXT),
            *"-MD -MP -MF".split(),
            "{dir}/.deps/binding.Tpo".format(dir=C_CODE),
            "{dir}/binding_{target}.cpp".format(dir=C_CODE, target=target),
            "-o", "{dir}/{mod}{pyext}".format(
                    dir=MODULES,
                    mod=module,
                    pyext=PYEXT
            ),
            "-DMODULE={}".format(module),
            *("-D{arg}".format(
                    arg=str(a).upper())
                    for a in args
            ),
            *("-D{key}={arg}".format(
                    key=str(kw).upper(), arg=str(a))
                    for kw, a in kwargs
            )
    ]
    print(" ".join(cmd[:2] + cmd[24:]))
    #print(" ".join(cmd))
    subprocess.call(cmd, cwd=C_CODE)

    return importlib.import_module(
            "GraphBLAS.modules.{mod}".format(mod=module)
    )


# dictionary that fetches missing modules by calling get_module()
class Cache(dict):

    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, params):

        target, args, kwargs = params

        # moduleID is only dependent on target and types
        module = dict.__getitem__(self, (
            target,
            tuple(sorted(args)),
            tuple(sorted(kwargs.items()))
        ))

        return module

    def __missing__(self, args):
        self[args] = get_module(*args)
        return dict.__getitem__(self, args)


cache = Cache()

