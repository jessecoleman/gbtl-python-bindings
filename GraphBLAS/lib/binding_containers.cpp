#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <graphblas/graphblas.hpp>
#include <graphblas/ComplementView.hpp>

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// matrix dtype
typedef DTYPE ScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<ScalarT> MatrixT;
typedef GraphBLAS::Vector<ScalarT> VectorT;

//typedef GraphBLAS::Matrix<bool> MMatrixT;
typedef GraphBLAS::MatrixComplementView<MatrixT> MatrixCompT;
//typedef GraphBLAS::Vector<bool> MVectorT;
typedef GraphBLAS::VectorComplementView<VectorT> VectorCompT;

MatrixT init_sparse_matrix(
        IndexT &rows, 
        IndexT &cols,
        IndexArrayT &i, 
        IndexArrayT &j, 
        std::vector<ScalarT> const &v); 

VectorT init_sparse_vector(
        IndexT &size,
        IndexArrayT &i, 
        std::vector<ScalarT> const &vals);

std::string print_matrix(MatrixT const &m);
std::string print_vector(VectorT const &v);

std::string print_matrix_comp(MatrixCompT const &m);
MatrixCompT matrix_complement(MatrixT const &m);
std::string print_vector_comp(VectorCompT const &v);
VectorCompT vector_complement(VectorT const &v);

std::vector<ScalarT> extract_vector(VectorT const v);

void define_matrix(py::module &m) 
{
    py::class_<MatrixT>(m, "Matrix")
        .def("__str__", &print_matrix, py::is_operator())
        .def("__invert__", &matrix_complement, py::is_operator())
        .def("nvals", &MatrixT::nvals)
        .def("nrows", &MatrixT::nrows)
        .def("ncols", &MatrixT::nrows);

    m.def("init_sparse_matrix", &init_sparse_matrix);
}

template <typename OtherVectorT>
void define_vector(py::module &m) 
{
    py::class_<VectorT>(m, "Vector")
        .def(py::init<OtherVectorT const &>())
        .def(py::init<std::vector<ScalarT> const &>())
        .def("__str__", &print_vector, py::is_operator())
        .def("__invert__", &vector_complement, py::is_operator())
        .def("nvals", &VectorT::nvals)
        .def("size", &VectorT::size)
        .def("__eq__", [](const VectorT &a, const VectorT &b) {
                return a == b;
        }, py::is_operator())
        .def("__ne__", [](const VectorT &a, const VectorT &b) {
                return a != b;
        }, py::is_operator());

    m.def("init_sparse_vector", &init_sparse_vector);
}

// initialize matrix and return it
MatrixT init_sparse_matrix(
        IndexT &rows, 
        IndexT &cols,
        IndexArrayT &i, 
        IndexArrayT &j, 
        std::vector<ScalarT> const &vals)
{
    MatrixT m(rows, cols);
    m.build(i, j, vals);
    return m;
}

// initialize vector and return it
VectorT init_sparse_vector(
        IndexT &size,
        IndexArrayT &i, 
        std::vector<ScalarT> const &vals) 
{
    VectorT v(size);
    v.build(i, vals);
    return v;
}

// return string representation of Matrix m
std::string print_matrix(MatrixT const &m) 
{
    std::ostringstream stream;
    GraphBLAS::print_matrix(stream, m); 
    return stream.str();
}

std::string print_vector(VectorT const &v) 
{
    std::ostringstream stream;
    GraphBLAS::print_vector<VectorT>(stream, v); 
    return stream.str();
}

MatrixCompT matrix_complement(MatrixT const &m) 
{ return GraphBLAS::complement(m); }


std::string print_matrix_comp(MatrixCompT const &m) 
{
    std::ostringstream stream;
    m.printInfo(stream); 
    return stream.str();
}

VectorCompT vector_complement(VectorT const &c) 
{ return GraphBLAS::complement(c); }

std::string print_vector_comp(VectorCompT const &v) 
{
    std::ostringstream stream;
    v.printInfo(stream); 
    return stream.str();
}

std::vector<ScalarT> extract_vector(VectorT v) {
    std::vector<ScalarT> result(v.size(),0);
    for (auto i = 0; i < v.size(); ++i) {
        if (v.hasElement(i)) { 
            result[i] = v.extractElement(i); 
        }
    }
    return result;
}

PYBIND11_MODULE(MODULE, m) {
    define_matrix(m);
    define_vector<VectorT>(m);

    py::class_<MatrixCompT>(m, "MatrixComplementView")
        .def("__str__", &print_matrix_comp);
    py::class_<VectorCompT>(m, "VectorComplementView")
        .def("__str__", &print_vector_comp);

#if MASK == 1
    py::class_<GraphBLAS::NoMask>(m, "NoMask")
        .def(py::init<>());
#endif
}
