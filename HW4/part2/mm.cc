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
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0){
        cin >> *n_ptr >> *m_ptr >> *l_ptr;
        *a_mat_ptr = (int*)malloc(sizeof(int) * *n_ptr * *m_ptr);
        *b_mat_ptr = (int*)malloc(sizeof(int) * *m_ptr * *l_ptr);

        int *ptr;
        for (int i = 0; i < *n_ptr; i++){
            for (int j = 0; j < *m_ptr; j++){
            ptr = *a_mat_ptr + i * *m_ptr + j;
                cin >> *ptr;
            }
        }

        for (int i = 0; i < *m_ptr; i++){
            for (int j = 0; j < *l_ptr; j++){
                ptr = *b_mat_ptr + i * *l_ptr + j;
                cin >> *ptr;
            }
        }
    }
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
    int size, rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int N, M, L;
    int rows;
    int offset;
    int i, j;
    if(rank == 0){
        N=n;
        M=m;
        L=l;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // master
    int send_type;
    if (rank == 0){
	    int *c;
    	c = (int*)malloc(sizeof(int) * n * l);
        // send data to slave
        int avg_row = n / (size - 1);
        offset = 0;
        for (int destination = 1; destination < size; destination++){
            rows = (destination <= (n % (size - 1)))? avg_row + 1: avg_row;
            send_type = 1;
            MPI_Send(&offset, 1, MPI_INT, destination, send_type, MPI_COMM_WORLD);
            send_type = 1;
            MPI_Send(&rows, 1, MPI_INT, destination, send_type, MPI_COMM_WORLD);
            send_type = 1;
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, destination, send_type, MPI_COMM_WORLD);
            send_type = 1;
            MPI_Send(&b_mat[0], m * l, MPI_INT, destination, send_type, MPI_COMM_WORLD);
            offset += rows;
        }
        
        // receive from slaves
        for (i = 1; i < size; i++){
            int src = i;
            send_type = 2;
            MPI_Recv(&offset, 1, MPI_INT, src, send_type, MPI_COMM_WORLD, &status);
            send_type = 2;
            MPI_Recv(&rows, 1, MPI_INT, src, send_type, MPI_COMM_WORLD, &status);
            send_type = 2;
            MPI_Recv(&c[offset * l], rows * l, MPI_INT, src, send_type, MPI_COMM_WORLD, &status);
        }

        // print result
        // for (i = 0; i < n; i++){
        //     for (j = 0; j < l; j++){
        // 	    cout << c[i * l + j];
        //         if (j != l-1) cout << " ";
        //     }
        //     cout << endl;
        // }
	    free(c);
    }
    // slave
    if (rank > 0){
	    int *a;
    	a = (int*)malloc(sizeof(int) * N * M);
    	int *b;
    	b = (int*)malloc(sizeof(int) * M * L);
    	int *c;
    	c = (int*)malloc(sizeof(int) * N * L);

        send_type = 1;
        MPI_Recv(&offset, 1, MPI_INT, 0, send_type, MPI_COMM_WORLD, &status);
        send_type = 1;
        MPI_Recv(&rows, 1, MPI_INT, 0, send_type, MPI_COMM_WORLD, &status);
        send_type = 1;
        MPI_Recv(&a[0], rows * M, MPI_INT, 0, send_type, MPI_COMM_WORLD, &status);
        send_type = 1;
        MPI_Recv(&b[0], M * L, MPI_INT, 0, send_type, MPI_COMM_WORLD, &status);

        for (int outer = 0; outer < L; outer++){
            for (i = 0; i < rows; i++){
                c[i * L + outer] = 0;
                for (j = 0; j < M; j++){
                    c[i * L + outer] += a[i * M + j] * b[j * L + outer];
                }
            }
        }
	    free(a);
    	free(b);

        send_type = 2;
        MPI_Send(&offset, 1, MPI_INT, 0, send_type, MPI_COMM_WORLD);
        send_type = 2;
        MPI_Send(&rows, 1, MPI_INT, 0, send_type, MPI_COMM_WORLD);
        send_type = 2;
        MPI_Send(&c[0], rows * L, MPI_INT, 0, send_type, MPI_COMM_WORLD);
	    free(c);
    }

}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0){
        delete [] a_mat;
        delete [] b_mat;
    }
}
