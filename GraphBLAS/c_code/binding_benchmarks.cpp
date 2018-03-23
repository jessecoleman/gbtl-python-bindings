#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <chrono>

#include <graphblas/graphblas.hpp>
//#include <graphblas.hpp>


namespace py = pybind11;
using namespace pybind11::literals;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// BFS types
#if defined(BFS)
#include <algorithms/bfs.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> GraphT;
#endif
#if defined(WAVEFRONT_TYPE) && defined(LEVELS_TYPE)
typedef GraphBLAS::Vector<WAVEFRONT_TYPE> WavefrontT;
typedef GraphBLAS::Vector<LEVELS_TYPE> LevelsT;
#endif


#if defined(MAXFLOW)
#include <algorithms/maxflow.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#ifdef RESULT_TYPE
typedef GraphBLAS::Vector<RESULT_TYPE> VectorT;
#else
typedef GraphBLAS::Vector<GRAPH_TYPE> VectorT;
#endif

#elif defined(METRICS)
#include <algorithms/metrics.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#ifdef RESULT_TYPE
typedef GraphBLAS::Vector<RESULT_TYPE> VectorT;
#else
typedef GraphBLAS::Vector<GRAPH_TYPE> VectorT;
#endif

#elif defined(MIS)
#include <algorithms/mis.hpp>
#elif defined(MST)
#include <algorithms/mst.hpp>
#elif defined(PAGE_RANK)
#include <algorithms/page_rank.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
typedef PAGE_RANK_TYPE RealT;
#elif defined(SSSP)
#include <algorithms/sssp.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
typedef GraphBLAS::Vector<PATH_TYPE> PathVectorT;

#elif defined(TRIANGLE_COUNT)
#include <algorithms/triangle_count.hpp>
#if defined(GRAPH_TYPE)
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#elif defined(L_TYPE)
typedef GraphBLAS::Matrix<L_TYPE> MatrixT;
#endif

#endif

#ifdef BFS
long benchmark_bfs(GraphT const  &graph,
                     WavefrontT   wavefront, //row vector, copy made
                     LevelsT     &levels)
{
    auto start = std::chrono::steady_clock::now();
    algorithms::bfs_level_masked_v2(graph, wavefront, levels);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);

    return duration.count();
}
#endif

#ifdef PAGE_RANK
long benchmark_page_rank(
        MatrixT const             &graph,
        GraphBLAS::Vector<RealT>  &page_rank,
        RealT                      damping_factor = 0.85,
        RealT                      threshold = 1.e-5,
        unsigned int max_iters = std::numeric_limits<unsigned int>::max())
{
    auto start = std::chrono::steady_clock::now();
    algorithms::page_rank(graph, page_rank, damping_factor, threshold, max_iters);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);

    return duration.count();
}
#endif

#ifdef SSSP
long benchmark_sssp(MatrixT const     &graph,
                      PathVectorT       &path)
 {
    auto start = std::chrono::steady_clock::now();
    algorithms::sssp(graph, path);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);

    return duration.count();
}
#endif

#ifdef TRIANGLE_COUNT
long benchmark_triangle_count(MatrixT const &graph)
 {
    auto start = std::chrono::steady_clock::now();
    auto cnt = algorithms::triangle_count_masked(graph);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);


    return duration.count();
}
#endif

#include <iostream>
#include <fstream>
#include <chrono>

#ifdef BUILD
py::tuple benchmark_build()//std::string pathname)
{
    typedef int64_t T;
    typedef GraphBLAS::Matrix<T> MatType;

    auto start = std::chrono::steady_clock::now();

    GraphBLAS::IndexArrayType iA, jA;
    int64_t num_rows = 0;
    int64_t max_id = 0;
    uint64_t src, dst;
    std::ifstream infile("graph.txt");
    while (infile)
    {
        infile >> src >> dst;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        if (src > max_id) max_id = src;
        if (dst > max_id) max_id = dst;

        iA.push_back(src);
        jA.push_back(dst);

        ++num_rows;
    }

    std::vector<T> v(iA.size(), 1);

    GraphBLAS::IndexType NUM_NODES(max_id + 1);

    auto duration1 = std::chrono::steady_clock::now();

    MatType A(NUM_NODES, NUM_NODES);
    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    auto duration2 = std::chrono::steady_clock::now();
    //std::cout << A << std::endl;

    unsigned int nnz = A.nvals();
    GraphBLAS::IndexArrayType i(nnz), j(nnz);
    std::vector<T> vout(nnz);
    A.extractTuples(i, j, vout);
    auto duration3 = std::chrono::steady_clock::now();

    //std::cout << i << j << std::endl;

    auto one = std::chrono::duration_cast<std::chrono::milliseconds>
        (duration1 - start);
    auto two = std::chrono::duration_cast<std::chrono::milliseconds>
        (duration2 - duration1);
    auto three = std::chrono::duration_cast<std::chrono::milliseconds>
        (duration3 - duration2);

    return py::make_tuple(one.count(), two.count(), three.count());
}

#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(BUILD)
    m.def("benchmark_build", &benchmark_build);//, "pathname"_a);
#endif

#if defined(BFS)
    m.def("benchmark_bfs", &benchmark_bfs, "graph"_a, "wavefront"_a, "levels"_a);
#elif defined(MAXFLOW)
    m.def("maxflow",    &algorithms::maxflow<MatrixT>, "capacity"_a, "source"_a, "sink"_a);
#elif defined(METRICS)
    m.def("vertex_in_degree",       &algorithms::vertex_in_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("vertex_out_degree",      &algorithms::vertex_out_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("vertex_degree",          &algorithms::vertex_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("graph_distance",         &algorithms::graph_distance<MatrixT, VectorT>, "graph"_a, "sid"_a, "result"_a);
    m.def("graph_distance_matrix",  &algorithms::graph_distance_matrix<MatrixT>, "graph"_a, "result"_a);
    m.def("vertex_eccentricity",    &algorithms::graph_distance_matrix<MatrixT>, "graph"_a, "vid"_a);
    m.def("graph_radius",           &algorithms::graph_radius<MatrixT>, "graph"_a);
    m.def("graph_diameter",         &algorithms::graph_diameter<MatrixT>, "graph"_a);
    m.def("closeness_centrality",   &algorithms::closeness_centrality<MatrixT>, "graph"_a, "vid"_a);
#elif defined(MIS)
    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>, "independent_set"_a);
    m.def("mis", &algorithms::mis<MatrixT>, "graph"_a, "independent_set"_a, "seed"_a);
#elif defined(MST)
    m.def("mst", &algorithms::mst<MatrixT>, "graph"_a, "mst_parents"_a);
#elif defined(PAGE_RANK)
    m.def("benchmark_page_rank", &benchmark_page_rank, 
            "graph"_a, 
            "page_rank"_a, 
            "damping_factor"_a = 0.85, 
            "threshold"_a = 1.e-5, 
            "max_iters"_a = std::numeric_limits<unsigned int>::max()
    );
#elif defined(SSSP)
    m.def("benchmark_sssp", &benchmark_sssp, "graph"_a, "path"_a);
//    m.def("batch_sssp", &algorithms::sssp<MatrixT, PathMatrixT>, "graph"_a, "paths"_a);
#elif defined(TRIANGLE_COUNT)
    m.def("benchmark_triangle_count_masked", &benchmark_triangle_count, "graph"_a);
    //m.def("triangle_count_masked", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame1", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame1a", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame2", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_newGBTL", &algorithms::triangle_count_newGBTL<LMatrixT, MatrixT>, "L"_a, "U"_a);
#endif
    
}
