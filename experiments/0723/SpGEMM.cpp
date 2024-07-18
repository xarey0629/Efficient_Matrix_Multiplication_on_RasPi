// gcc -fopenmp -o SpGEMM SpGEMM.cpp && ./SpGEMM 128 0.25
// g++ -fopenmp -o SpGEMM SpGEMM.cpp && ./SpGEMM 128 0.25


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ctime>
#include "BIN.h"

// #define RATIO   0.25    // Sparse density ratio
#define SHOW    0       // 0: Do not show matrices. 1: Show matrices.

// Global variables
int randCnt = 0;        // Random number generator counter
float RATIO = 0.25;    // Sparse density ratio

// Get time in seconds
double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec; // tv.tv_usec is the number of microsecond since last second.
}

// Generate a sparse matrix with random values.
// The number of non-zero elements is RATIO * rows * cols
void prepareSparse(float* matrix, int rows, int cols){
    int nnz = (int)(rows * cols * RATIO);
    // printf("Number of non-zero elements: %d\n", nnz);
    srand((unsigned)time(NULL) + (randCnt++));
    for(int h = 0; h < nnz; h++) {
      int i = rand() % rows;
      int j = rand() % cols;
      if(matrix[i * cols + j] != 0){
        h--; continue;
      }
      matrix[i * cols + j] = (float)(rand() % 20000 - 10000) / 1000; // Random number between -10 and 10.
    }
}

// Convert a matrix to CSR format
void matrixToCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    int nnzCnt = 0;
    for(int row = 0; row < rows; row++){
        rowPtr[row] = nnzCnt;
        for(int col = 0; col < cols; col++){
            float num = matrix[row * cols + col];
            if(num != 0){
                val[nnzCnt] = num;
                colInd[nnzCnt] = col;
                nnzCnt++;
            }
        }
    }
    rowPtr[rows] = nnzCnt; // End of row pointer
}

// Convert CSR back to a normal matrix. 
// Not used yet.
void CSRToMatrix(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    for(int i = 0; i < rows; i++){
        int row_start = rowPtr[i];
        int row_end = rowPtr[i+1];
        for(int idx = row_start; idx < row_end; idx++){
            matrix[i * cols + colInd[idx]] = val[idx];
        }
    }
}

// Display a matrix
void showMatrix(float* matrix, int rows, int cols){
    if(SHOW == 0) return; // If SHOW is 0, do not show the matrix.
    
    printf("Matrix: \n");
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Display a matrix in CSR format
void showCSR(int rows, int cols, int* rowPtr, int *colInd, float* val){
    if(SHOW == 0) return; // If SHOW is 0, do not show the matrix.
    
    printf("CSR: \n");
    printf("rowPtr: ");
    for(int i = 0; i < rows + 1; i++){
        printf("%d ", rowPtr[i]);
    }
    printf("\n");

    printf("colInd: ");
    // int n = RATIO * rows * cols;
    int n = rowPtr[rows];
    for(int i = 0; i < n; i++){
        printf("%d ", colInd[i]);
    }
    printf("\n");

    printf("val: ");
    for(int i = 0; i < n; i++){
        printf("%.1f ", val[i]);
    }
    printf("\n\n");
}

// Input a pointer of a matrix and zeroize it
void zeroizeMatrix(float* matrix, int rows, int cols){
    printf("******** Zeroize Matrix ********\n");
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matrix[i * cols + j] = 0;
        }
    }
}

// Input two matrices and print the multiplication result
void simpleMatrixMultiplication(float* matrixA, float* matrixB, float* matrixC, 
                            int M, int K, int N){

    printf("********** ********** **********\n");
    printf("Simple Multiplicaiton: \n");
    printf("- M x K x N = %d x %d x %d\n", M, K, N);
    printf("- Number of non-zero elements in A: %d, in B: %d\n", (int)(M * K * RATIO), (int)(K * N * RATIO));
    double start = get_time();
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            matrixC[i * N + j] = 0;
            for(int k = 0; k < K; k++){
                matrixC[i * N + j] += matrixA[i * K + k] * matrixB[k * N + j];
            }
        }
    }
    double end = get_time();
    printf("- Time: %.3f\n", end - start);
    
    showMatrix(matrixC, M, N);
    printf("********** ********** **********\n");
}

// Gustavson's algorithm
void sparse_matrix_mul(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                float* matrixC, int M, int K, int N){ 
    
    printf("********** ********** **********\n");
    printf("Sparse Multiplication: \n");
    printf("- M x K x N = %d x %d x %d\n", M, K, N);
    printf("- Number of non-zero elements in A: %d, in B: %d\n", (int)(M * K * RATIO), (int)(K * N * RATIO));
    
    double start = get_time();
    // #pragma omp parallel for num_threads(threads)
    for(int m = 0; m < M; m++){
        int row_start1 = rowPtr1[m];
        int row_end1 = rowPtr1[m + 1];
        for(int idx1 = row_start1; idx1 < row_end1; idx1++){
            int col_1 = colInd1[idx1];            
            for(int idx2 = rowPtr2[col_1]; idx2 < rowPtr2[col_1 + 1]; idx2++){
                // Size of matrixC is M x N
                matrixC[m * N + colInd2[idx2]] += val1[idx1] * val2[idx2];
            }
        }
    }
    double end = get_time();
    printf("- Time: %.3f\n", end - start);
    
    showMatrix(matrixC, M, N);
    printf("********** ********** **********\n");
}


void single_threaded_hashing_SpGEMM(const int *rowPtrA, const int *colIndA, const float *valA, 
                            const int *rowPtrB, const int *colIndB, const float *valB, 
                            int *&c_rowPtr, int *&c_colInd, float *&c_val, const int M, const int N,
                            int num_of_threads = 1){
    double start = get_time();
    execute_hashing_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N, num_of_threads);
    double end = get_time();
    printf("Single-threaded Hashing SpGEMM --> Time: %.3f\n", end - start);
    showCSR(M, N, c_rowPtr, c_colInd, c_val);
}

void multi_threaded_hashing_SpGEMM(const int *rowPtrA, const int *colIndA, const float *valA, 
                            const int *rowPtrB, const int *colIndB, const float *valB, 
                            int *&c_rowPtr, int *&c_colInd, float *&c_val, const int M, const int N,
                            int num_of_threads = omp_get_max_threads()){
    double start = get_time();
    execute_hashing_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N, num_of_threads);
    double end = get_time();
    printf("Multi-threaded Hashing SpGEMM--> Time: %.3f\n", end - start);
    showCSR(M, N, c_rowPtr, c_colInd, c_val);
}

int main(int argc, char *argv[]){
    // ************** Input(argv): M, K, N **************
    int M = strtol(argv[1], NULL, 10);
    int N = M, K = M;
    RATIO = strtof(argv[2], NULL);
    // int K = strtol(argv[2], NULL, 10);
    // int N = strtol(argv[3], NULL, 10);
    
    // ************** Multiplication ************** 
    //            C   =    A    x    B
    //          M x N = (M x K) x (K x N)

    // ************** Memory Allocation **************
    float *matrixA = (float*)malloc(M * K * sizeof(float));
    int nnzA = (int)(M * K * RATIO);
    int *rowPtrA = (int*)malloc((M + 1) * sizeof(int));
    int *colIndA = (int*)malloc(nnzA * sizeof(int));
    float *valA = (float*)malloc(nnzA * sizeof(float));

    float *matrixB = (float*)malloc(K * N * sizeof(float));
    int nnzB = (int)(K * N * RATIO);
    int *rowPtrB = (int*)malloc((K + 1) * sizeof(int));
    int *colIndB = (int*)malloc(nnzB * sizeof(int));
    float *valB = (float*)malloc(nnzB * sizeof(float));
    
    // Produce random elements in sparse matrix A and B.
    prepareSparse(matrixA, M, K);
    prepareSparse(matrixB, K, N);

    // Display matrix A and B, and their CSR format.
    showMatrix(matrixA, M, K);
    matrixToCSR(matrixA, M, K, rowPtrA, colIndA, valA);
    showCSR(M, K, rowPtrA, colIndA, valA);

    showMatrix(matrixB, K, N);
    matrixToCSR(matrixB, K, N, rowPtrB, colIndB, valB);
    showCSR(K, N, rowPtrB, colIndB, valB);

    // Allocate matrix C
    float *matrixC = (float*)malloc(M * N * sizeof(float));

    // ************** Multiplication Execution **************

    // Execute simple matrix multiplication O(n^3)
    // simpleMatrixMultiplication(matrixA, matrixB, matrixC, M, K, N);
    // Reset matrix C
    // zeroizeMatrix(matrixC, M, N);
    // Execute sparse matrix multiplication O(flops)
    sparse_matrix_mul(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, matrixC, M, K, N);
   
    // ************** Parallel Hashing SpGEMM **************
    int *c_rowPtr, *c_colInd;
    float *c_val;
    // int num_of_threads;

    // Single-threaded
    // single_threaded_hashing_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N);

    // Multi-threaded
    multi_threaded_hashing_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N);

    // ************** SpArr SpGEMM **************
    double start = get_time();
    execute_spArr_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N, false);
    double end = get_time();
    printf("SpArr SpGEMM (NEON OFF) --> Time: %.3f\n", end - start);
    showCSR(M, N, c_rowPtr, c_colInd, c_val);

    start = get_time();
    execute_spArr_SpGEMM(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, c_rowPtr, c_colInd, c_val, M, N, true);
    end = get_time();
    printf("SpArr SpGEMM (NEON ON) --> Time: %.3f\n", end - start);
    showCSR(M, N, c_rowPtr, c_colInd, c_val);
    
    return 0;
}