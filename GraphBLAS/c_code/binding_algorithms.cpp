#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// matrix storage type
typedef GraphBLAS::Matrix<ATYPE> AMatrixT;
typedef GraphBLAS::Matrix<BTYPE> BMatrixT;
typedef GraphBLAS::Matrix<CTYPE> CMatrixT;
// vector storage type
typedef GraphBLAS::Vector<ATYPE> UVectorT;
typedef GraphBLAS::Vector<BTYPE> VVectorT;
typedef GraphBLAS::Vector<CTYPE> WVectorT;

#if defined(BFS)
#include <algorithms/bfs.hpp>
#elif defined(MAXFLOW)
#include <algorithms/maxflow.hpp>
#elif defined(METRICS)
#include <algorithms/metrics.hpp>
#elif defined(MIS)
#include <algorithms/mis.hpp>
#elif defined(MST)
#include <algorithms/mst.hpp>
#elif defined(PAGERANK)
#include <algorithms/page_rank.hpp>
#elif defined(SSSP)
#include <algorithms/sssp.hpp>
#elif defined(TRICOUNT)
#include <algorithms/tricount.hpp>
#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(BFS)
    m.def("bfs", &algorithms::bfs<AMatrixT, VVectorT, WVectorT>);
    m.def("bfs_batch", &algorithms::bfs_batch<AMatrixT, BMatrixT, CMatrixT>);
    m.def("bfs_level", &algorithms::bfs_level<AMatrixT, BMatrixT, CMatrixT>);
#elif defined(MAXFLOW)
    m.def("push", &algorithms::push<MatrixT, VectorT>);
    m.def("relabel", &ralgorithms::elabel<MatrixT, VectorT>);
    m.def("discharge", &dalgorithms::ischarge<MatrixT, VectorT>);
    m.def("maxflow", &malgorithms::axflow<MatrixT>);
#elif defined(METRICS)
    m.def("vertex_in_degree", &algorithms::vertex_in_degree<AMatrixT>);
    m.def("vertex_out_degree", &algorithms::vertex_out_degree<AMatrixT>);
    m.def("vertex_degree", &algorithms::vertex_degree<AMatrixT>);
    m.def("graph_distance", &algorithms::graph_distance<AMatrixT, UVectorT>);
    m.def("graph_distance_matrix", &algorithms::graph_distance_matrix<AMatrixT>);
    m.def("vertex_eccentricity", &algorithms::graph_distance_matrix<AMatrixT>);
    m.def("graph_radius", &algorithms::graph_radius<AMatrixT>);
    m.def("graph_diameter", &algorithms::graph_diameter<AMatrixT>);
    m.def("closeness_centrality", &algorithms::closeness_centrality<AMatrixT>);
#elif defined(MIS)
    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>);
    m.def("mis", &algorithms::mis<AMatrixT>);
#elif defined(MST)
    m.def("mst", &algorithms::mst<AMatrixT>);
#elif defined(PAGERANK)
    m.def("page_rank", &page_rank<AMatrixT, BMatrixT>);
#elif defined(SSSP)
    m.def("sssp", &algorithms::sssp<AMatrixT, UVectorT>);
#elif defined(TRICOUNT)
    m.def("triangle_count", &algorithms::triangle_count<AMatrixT>);
    m.def("triangle_count_newGBTL", &algorithms::triangle_count_newGBTL<AMatrixT, BMatrixT>);
#endif
}
