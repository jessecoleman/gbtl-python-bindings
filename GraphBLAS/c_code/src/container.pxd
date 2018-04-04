from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "graphblas/graphblas.hpp" namespace "GraphBLAS":
    cdef cppclass Matrix[T]:

        Matrix(int rows, int cols)
        void build(vector[int].iterator &i,
                   vector[int].iterator &j,
                   vector[T].iterator &vals,
                   int num_vals)
        int nvals()
        int nrows()
        int ncols()
        bool hasElement(int row, int col)
        void setElement(int row, int col, T val)
        T extractElement(int row, int col)
        # TODO extractTuples()
        void transpose() 
        void matrix_complement()
        bool operator==()

    cdef cppclass MatrixTransposeView[T]:

        int nvals()
        int nrows()
        int ncols()
        bool hasElement(int row, int col)
        T extractElement(int row, int col)
        bool operator==()


    cdef cppclass MatrixComplementView[T]:

        int nvals()
        int nrows()
        int ncols()
        bool hasElement(int row, int col)
        T extractElement(int row, int col)
        bool operator==()

    cdef cppclass Vector[T]:

        Vector(int size)
        void build(vector[int].iterator &i,
                   vector[T].iterator &vals,
                   int num_vals)
        int nvals()
        int size()
        bool hasElement(int index)
        void setElement(int index, T val)
        T extractElement(int index)
        # TODO extractTuples()
        bool operator==()


    cdef cppclass VectorComplementView[T]:

        void build(vector[int].iterator &i,
                   vector[T].iterator &vals,
                   int num_vals)
        int nvals()
        int size()
        bool hasElement(int index)
        void setElement(int index, T val)
        T extractElement(int index)
        # TODO extractTuples()
        bool operator==()


    cdef cppclass NoAccumulate: 
        NoAccumulate()
    
    cdef cppclass ArithmeticSemiring[T]: 
        ArithmeticSemiring()

    cdef void mxm[
        CMatrixT, 
        MaskT, 
        AccumT, 
        SemiringT, 
        AMatrixT, 
        BMatrixT](
            CMatrixT &C,
            const MaskT &M,
            AccumT accum,
            SemiringT op,
            const AMatrixT &A,
            const BMatrixT &B,
            bool replace_flag)

