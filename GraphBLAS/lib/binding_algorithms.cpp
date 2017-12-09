#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// matrix dtype
typedef ATYPE AScalarT;
typedef BTYPE BScalarT;
typedef CTYPE CScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<AScalarT> AMatrixT;
typedef GraphBLAS::Matrix<BScalarT> BMatrixT;
typedef GraphBLAS::Matrix<CScalarT> CMatrixT;
// vector storage type
typedef GraphBLAS::Vector<AScalarT> UVectorT;
typedef GraphBLAS::Vector<BScalarT> VVectorT;
typedef GraphBLAS::Vector<CScalarT> WVectorT;

#ifdef MAXFLOW
#include <algorithms/maxflow.hpp>
template <typename MatrixT, typename VectorT>
void define_maxflow_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("push", &push<MatrixT, VectorT>);
    m.def("relabel", &relabel<MatrixT, VectorT>);
    m.def("discharge", &discharge<MatrixT, VectorT>);
    m.def("maxflow", &maxflow<MatrixT>);
}
#endif

#ifdef METRICS
#include <algorithms/metrics.hpp>
template <typename MatrixT, typename VectorT>
void define_metrics_algorithms(py::module &m) {
    m.def("vertex_in_degree", &algorithms::vertex_in_degree<MatrixT>);
    m.def("vertex_out_degree", &algorithms::vertex_out_degree<MatrixT>);
    m.def("vertex_degree", &algorithms::vertex_degree<MatrixT>);
    m.def("graph_distance", &algorithms::graph_distance<MatrixT, VectorT>);
    m.def("graph_distance_matrix", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("vertex_eccentricity", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("graph_radius", &algorithms::graph_radius<MatrixT>);
}
#endif

#ifdef MIS
#include <algorithms/mis.hpp>
template <typename T, typename MatrixT>
void define_mis_algorithms(py::module &m) {
    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>);
    m.def("mis", &algorithms::mis<MatrixT>);
}
#endif
 
#ifdef MST
#include <algorithms/mst.hpp>
template <typename T, typename MatrixT>
void define_mst_algorithms(py::module &m) {
    //py::class_<LessThan<T>>(m, "LessThan");

    using namespace algorithms;
    m.def("mst", &mst<MatrixT>);
}
#endif

#ifdef PAGERANK
#include <algorithms/page_rank.hpp>
template <typename MatrixT, typename PRMatrixT>
void define_pagerank_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("page_rank", &page_rank<MatrixT, PRMatrixT>);
}
#endif

#ifdef SSSP
#include <algorithms/sssp.hpp>
template <typename MatrixT,
          typename VectorT>
void define_sssp_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("sssp", &sssp<MatrixT, VectorT>);
}
#endif

#ifdef TRICOUNT
#include <algorithms/triangle_count.hpp>
template <typename MatrixT>
void define_tri_count_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("triangle_count", &triangle_count<MatrixT>);
    m.def("triangle_count_newGBTL", &triangle_count_newGBTL<MatrixT, MatrixT>);
}
#endif

#ifdef BFS
#include <algorithms/bfs.hpp>
#endif

PYBIND11_MODULE(MODULE, m) {
#ifdef BFS
    m.def("bfs", &algorithms::bfs<AMatrixT, VVectorT, WVectorT>);
    m.def("bfs_batch", &algorithms::bfs_batch<AMatrixT, BMatrixT, CMatrixT>);
    m.def("bfs_level", &algorithms::bfs_level<AMatrixT, BMatrixT, CMatrixT>);
#endif
#ifdef MAXFLOW
    define_maxflow_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef METRICS
    define_metrics_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef MIS
    define_mis_algorithms<ScalarT, GBMatrix>(m);
#endif
#ifdef MST
    define_mst_algorithms<ScalarT, GBMatrix>(m);
#endif
#ifdef SSSP
    define_sssp_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef TRICOUNT
    define_tri_count_algorithms<GBMatrix>(m);
#endif
}
