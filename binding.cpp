#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "graphblas/graphblas.hpp"
#include "algorithms/algorithms.hpp"

namespace py = pybind11;

// matrix data type
typedef double ScalarT;
// matrix index type
typedef graphblas::IndexType IndexT;
//
typedef graphblas::IndexArrayType IndexArrayT;
// matrix storage type
typedef graphblas::Matrix<ScalarT, graphblas::DirectedMatrixTag> GBMatrix;
// vector storage type
typedef graphblas::Vector<ScalarT> GBVector;

template <typename MatrixT,
          typename ScalarT, 
          typename IndexT>
void build_matrix(std::vector<ScalarT> i, std::vector<ScalarT> j, std::vector<ScalarT> v, IndexT n);

template <typename MatrixT,
          typename ScalarT, 
          typename IndexT>
void build_matrix(MatrixT &m, std::vector<ScalarT> i, std::vector<ScalarT> j, std::vector<ScalarT> v, IndexT n);

template <typename MatrixT>
std::string print_matrix(MatrixT &m);

template <typename MatrixT, 
          typename IndexT, 
          typename ScalarT>
void define_matrix(py::module &m) {
    typedef graphblas::IndexArrayType::iterator Iterator;
    typedef graphblas::math::Assign<ScalarT> AccumT;

    py::class_<MatrixT>(m, "Matrix")
        .def(py::init<IndexT, IndexT, ScalarT>())
        .def(py::init<std::vector<std::vector<ScalarT>> const &, ScalarT>())
        .def(py::init<const MatrixT &>())
        .def("get_shape", py::overload_cast<>(&MatrixT::get_shape, py::const_))
        .def("get_shape", py::overload_cast<IndexT &, IndexT &>(&MatrixT::get_shape, py::const_))
        .def("get_zero", &MatrixT::get_zero)
        .def("set_zero", &MatrixT::set_zero)
        .def("get_nnz", &MatrixT::get_nnz)
        .def("__str__", &print_matrix<MatrixT>, py::is_operator());
        //.def("build_matrix", py::overload_cast<std::vector<ScalarT>, std::vector<ScalarT>, std::vector<ScalarT>, IndexT>
        //        (&build_matrix<MatrixT, ScalarT, IndexT>));

        //.def(py::this == MatrixT())
        //.def(py::this != MatrixT())
        //.def("__getitem__", [](const py::tuple &args) { 
        //    return MatrixT::get_value_at(x, y); 
        //}, py::is_operator())
        //.def("print_info", &MatrixT::print_info);

    m.def("build_matrix", py::overload_cast<MatrixT &, std::vector<ScalarT>, std::vector<ScalarT>, std::vector<ScalarT>, IndexT>
            (&build_matrix<MatrixT, ScalarT, IndexT>));
    m.def("identity", &graphblas::identity<MatrixT>);
}

template <typename MatrixT,
          typename ScalarT, 
          typename IndexT>
void build_matrix(std::vector<ScalarT> i, std::vector<ScalarT> j, std::vector<ScalarT> v, IndexT n) {
    std::cout << "Testing1" << std::endl;
    for (auto it = i.begin(); it != i.end(); ++it)
        std::cout << *it << ", ";
    graphblas::Matrix<MatrixT>::buildmatrix(i.begin(), j.begin(), v.begin(), n);
}

template <typename MatrixT,
          typename ScalarT, 
          typename IndexT>
void build_matrix(MatrixT &m, std::vector<ScalarT> i, std::vector<ScalarT> j, std::vector<ScalarT> v, IndexT n) {
    for (auto it = i.begin(); it != i.end(); ++it)
        std::cout << *it << ", ";
    graphblas::buildmatrix(m, i.begin(), j.begin(), v.begin(), n);
}

template <typename MatrixT>
std::string print_matrix(MatrixT &m) {
    std::ostringstream stream;
    graphblas::print_matrix(stream, m); 
    return stream.str();
}

// TODO
//template <typename MatrixT,
//          typename ScalarT, 
//          typename AccumT>
//void build_matrix(MatrixT &m, std::vector<ScalarT> const &i, std::vector<ScalarT> const &j, std::vector<ScalarT> const &v, AccumT accum) {
//    buildmatrix(m, i.begin(), j.begin(), v.begin(), n, accum);
//}

//typedef graphblas::Vector<ScalarT, graphblas::
template <typename VectorT, 
          typename OtherVectorT, 
          typename SizeT, 
          typename ScalarT>
void define_vector(py::module &m) {
    py::class_<VectorT>(m, "Vector")
        .def(py::init<OtherVectorT const &>())
        .def(py::init<SizeT const &, ScalarT const &>())
        .def("__eq__", [](const VectorT &a, const VectorT &b) {
                return a == b;
        }, py::is_operator())
        .def("__ne__", [](const VectorT &a, const VectorT &b) {
                return a != b;
        }, py::is_operator());
}

template <typename MatrixT, 
          typename ClusterMatrixT>
void define_cluster_algorithms(py::module &m) {
    m.def("get_cluster_assignments_v2", &algorithms::get_cluster_assignments_v2<MatrixT>);
    m.def("get_cluster_assignments", &algorithms::get_cluster_assignments<MatrixT>);
    m.def("peer_pressure_cluster", &algorithms::peer_pressure_cluster<MatrixT, ClusterMatrixT>);
    m.def("peer_pressure_cluster", &algorithms::peer_pressure_cluster<MatrixT>);
    m.def("peer_pressure_cluster_v2", &algorithms::peer_pressure_cluster_v2<MatrixT, ClusterMatrixT>);
    m.def("markov_cluster", &algorithms::markov_cluster<MatrixT>);
}

template <typename MatrixT, 
          typename WavefrontMatrixT, 
          typename ParentListMatrixT, 
          typename LevelListMatrixT>
void define_bfs_algorithms(py::module &m) {
    m.def("bfs", &algorithms::bfs<MatrixT, WavefrontMatrixT, ParentListMatrixT>);
    m.def("bfs_level", &algorithms::bfs_level<MatrixT, WavefrontMatrixT, ParentListMatrixT>);
    m.def("bfs_level_masked", &algorithms::bfs_level_masked<MatrixT, WavefrontMatrixT, LevelListMatrixT>);
}

template <typename MatrixT>
void define_maxflow_algorithms(py::module &m) {
    m.def("push", &push<MatrixT>);
    m.def("relabel", &relabel<MatrixT>);
    m.def("discharge", &discharge<MatrixT>);
    m.def("maxflow", &algorithms::maxflow<MatrixT>);
}

template <typename MatrixT>
void define_metrics_algorithms(py::module &m) {
    m.def("vertex_count", &algorithms::vertex_count<MatrixT>);
    m.def("edge_count", &algorithms::edge_count<MatrixT>);
    m.def("vertex_in_degree", &algorithms::vertex_in_degree<MatrixT>);
    m.def("vertex_out_degree", &algorithms::vertex_out_degree<MatrixT>);
    m.def("vertex_degree", &algorithms::vertex_degree<MatrixT>);
    m.def("graph_distance", &algorithms::graph_distance<MatrixT>);
    m.def("graph_distance_matrix", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("vertex_eccentricity", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("graph_radius", &algorithms::graph_radius<MatrixT>);
}

template <typename T, typename MatrixT>
void define_mis_algorithms(py::module &m) {
    //py::class_<SetRandom>(m, "SetRandom")
    //    .def(py::init<>());
    
    //py::class_<GreaterThan<T>>(m, "GreaterThan")
    //    .def(py::init<>());

    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>);
    m.def("mis", &algorithms::mis<MatrixT>);
}

template <typename T, typename MatrixT>
void define_mst_algorithms(py::module &m) {
    //py::class_<LessThan<T>>(m, "LessThan");

    using namespace algorithms;
    m.def("mst", &mst<MatrixT>);
}

template <typename MatrixT, 
          typename PRMatrixT>
void define_pagerank_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("page_rank", &page_rank<MatrixT, PRMatrixT>);
}

template <typename MatrixT, 
          typename OtherMatrixT, 
          typename IndexT,
          typename SemiringT, 
          typename BackendT>
void define_view(py::module &m) {
    //using namespace graphblas;
    typedef graphblas::TransposeView<MatrixT> TransposeView;
    py::class_<TransposeView>(m, "TransposeView")
        .def(py::init<BackendT>)
        .def("get_shape", py::overload_cast<IndexT &, IndexT &>(&TransposeView::get_shape, py::const_))
        .def("get_shape", py::overload_cast<>(&TransposeView::get_shape, py::const_))
        .def("get_zero", &TransposeView::get_zero)
        .def("get_nnz", &TransposeView::get_nnz)
        .def("print_info", &TransposeView::print_info)
        .def(py::self == MatrixT())
        .def(py::self != MatrixT());

    typedef graphblas::NegateView<MatrixT, SemiringT> NegateView;
    py::class_<NegateView>(m, "NegateView")
        .def(py::init<BackendT>())
        .def(py::init<NegateView>())
        .def("get_shape", py::overload_cast<IndexT &, IndexT &>(&NegateView::get_shape)) 
        .def("get_shape", py::overload_cast<>(&NegateView::get_shape)) 
        .def("get_zero", &NegateView::get_zero)
        .def("get_nnz", &NegateView::get_nnz)
        .def("get_value_at", &NegateView::get_value_at)
        .def("print_info", &NegateView::print_info)
        .def(py::self == MatrixT())
        .def(py::self != MatrixT());
}


PYBIND11_MODULE(graphBLAS, m) {
    define_matrix<GBMatrix, IndexT, ScalarT>(m);
    define_vector<GBVector, std::vector<ScalarT>, IndexT, ScalarT>(m);

    ScalarT default_inf(std::numeric_limits<ScalarT>::max());
    py::object max_val = py::cast(default_inf);
    m.attr("max_val") = max_val;

    // define algorithms
    py::module algorithms_m = m.def_submodule("algorithms", "algorithms for use on matrices");

    define_cluster_algorithms<GBMatrix, GBMatrix>(algorithms_m);
    define_bfs_algorithms<GBMatrix, GBMatrix, GBMatrix, GBMatrix>(algorithms_m);
    define_maxflow_algorithms<GBMatrix>(algorithms_m);
    define_metrics_algorithms<GBMatrix>(algorithms_m);
    define_mis_algorithms<ScalarT, GBMatrix>(algorithms_m);
    define_mst_algorithms<ScalarT, GBMatrix>(algorithms_m);
    //define_pagerank_algorithms<GBMatrix, GBMatrix>(algorithms_m);
}
