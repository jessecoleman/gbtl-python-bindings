#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <graphblas/graphblas.hpp>
//#include <graphblas.hpp>

namespace py = pybind11;

    class IndexOutOfBoundsException : public std::exception
    class PanicException : public std::exception
    class InvalidValueException : public std::exception
    class InvalidIndexException : public std::exception
    class DimensionException : public std::exception
    class OutputNotEmptyException : public std::exception
    class NoValueException : public std::exception

