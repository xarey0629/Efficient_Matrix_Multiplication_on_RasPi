// gcc -fopenmp -o SpGEMM SpGEMM.cpp && ./SpGEMM 128
// g++ -fopenmp -o SpGEMM SpGEMM.cpp && ./SpGEMM 128


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "BIN.h"

#define RATIO   0.25    // Sparse density ratio
#define SHOW    0       // 0: Do not show matrices. 1: Show matrices.

// Global variables
int randCnt = 0;        // Random number generator counter

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

// Convert CSR format back to a normal matrix. 
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
    int n = RATIO * rows * cols;
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



int main(int argc, char *argv[]){
    // ************** Input(argv): M, K, N **************
    int M = strtol(argv[1], NULL, 10);
    int N = M, K = M;
    // int K = strtol(argv[2], NULL, 10);
    // int N = strtol(argv[3], NULL, 10);
    
    // ************** Multiplication ************** 
    //            C   =    A    x    B
    //          M x N = (M x K) x (K x N)
    
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

    // Test BIN
    BIN myBin(M, 8);
    myBin.set_intprod_num(rowPtrA, colIndA, rowPtrB, M);
    printf("Total intprod: %lld\n", myBin.total_intprod);
    // print each row's nnz
    for(int i = 0; i < M; i++){
        printf("row_nnz[%d]: %d\n", i, myBin.row_nnz[i]);
    }
    myBin.set_rows_offset(M);
    // print row_offset
    for(int i = 0; i < myBin.num_of_threads + 1; i++){
        printf("row_offset[%d]: %d\n", i, myBin.row_offset[i]);
    }
    
    



    

    // Create matrix C
    float *matrixC = (float*)malloc(M * N * sizeof(float));
    simpleMatrixMultiplication(matrixA, matrixB, matrixC, M, K, N);
    zeroizeMatrix(matrixC, M, N);
    sparse_matrix_mul(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, matrixC, M, K, N);

    return 0;
}