#include <mpi.h>
#include <cstdio>
#include<iostream>

using namespace std;

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    cin >> *n_ptr >> *m_ptr >> *l_ptr;

    a_mat_ptr = new int*[*n_ptr];
    for (int i = 0; i < *n_ptr; ++i) {
        a_mat_ptr[i] = new int[*m_ptr];
    }

    for (int i = 0; i < *n_ptr; ++i) {
        for (int j = 0; j < *m_ptr; ++j) {
            cin >> a_mat_ptr[i][j];
        }
    }

    b_mat_ptr = new int*[*m_ptr];
    for (int i = 0; i < *m_ptr; ++i) {
        b_mat_ptr[i] = new int[*l_ptr];
    }

    for (int i = 0; i < *m_ptr; ++i) {
        for (int j = 0; j < *l_ptr; ++j) {
            cin >> b_mat_ptr[i][j];
        }
    }

    // print matrix data
    // cout << "Matrix data:\n";
    // for (int i = 0; i < *n_ptr; ++i) {
    //     for (int j = 0; j < *m_ptr; ++j) {
    //         cout << a_mat_ptr[i][j] << " ";
    //     }
    //     cout << "\n";
    // }

    // 释放内存
    // for (int i = 0; i < rows; ++i) {
    //     delete[] a_mat_ptr[i];
    // }
    // delete[] a_mat_ptr;
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{

}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank!=0)return;
    delete [] a_mat;
    delete [] b_mat;
}

// int main () {
//     int n, m, l;
//     int *a_mat, *b_mat;

//     MPI_Init(NULL, NULL);
//     double start_time = MPI_Wtime();

//     construct_matrices(&n, &m, &l, &a_mat, &b_mat);
//     matrix_multiply(n, m, l, a_mat, b_mat);
//     destruct_matrices(a_mat, b_mat);

//     double end_time = MPI_Wtime();
//     MPI_Finalize();
//     printf("MPI running time: %lf Seconds\n", end_time - start_time);

//     return 0;
// }
