#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define RATIO 0.25

int randCnt = 0;

void prepareSparse(float* matrix, int rows, int cols){
    int nnz = RATIO * rows * cols;
    srand((unsigned)time(NULL) + (randCnt++));
    for(int h = 0; h < nnz; h++) {
      int i = rand() % rows;
      int j = rand() % cols;
      if(matrix[i * cols + j] != 0){
          h--;
          continue;
      }
      matrix[i * cols + j] = (float)(rand() % 20000 - 10000)/1000;
    }
    // toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
}

void matrixToCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row = 0; row < rows; row++){
        rowPtr[row] = size;
        for(int col = 0; col < cols; col++){
            float num = matrix[row * cols + col];
            if(num != 0){
                val[size] = num;
                colInd[size] = col;
                size++;
            }
        }
    }
    rowPtr[rows] = size;
}

void CSRToMatrix(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    for(int i = 0; i < rows; i++){
        int row_start = rowPtr[i];
        int row_end = rowPtr[i+1];
        for(int idx = row_start; idx < row_end; idx++){
            matrix[i * cols + colInd[idx]] = val[idx];
        }
    }
}

void showMatrix(float* matrix, int rows, int cols){
    printf("Matrix: \n");
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void showCSR(int rows, int cols, int* rowPtr, int *colInd, float* val){
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

void sparse_matrix_mul3(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                float* matrixC, int M, int K, int N, int threads){ 

    // #pragma omp parallel for num_threads(threads)
    for(int m = 0; m < M; m++){
        int row_start1 = rowPtr1[m];
        int row_end1 = rowPtr1[m + 1];
        for(int idx = row_start1; idx < row_end1; idx++){
            int k = colInd1[idx];
            for(int t = rowPtr2[k]; t < rowPtr2[k + 1]; t++){
                matrixC[m * N + colInd2[t]] += val1[idx] * val2[t];
            }
        }
    }
    
}

int main(int argc, char *argv[]){
    // Input: M, K, N
    // A: M x K
    // B: K x N
    // C: M x N
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);

    float *matrixA = (float*)malloc(M * K * sizeof(float));
    int *rowPtrA = (int*)malloc((M + 1) * sizeof(int));
    int *colIndA = (int*)malloc(M * K * RATIO * sizeof(int));
    float *valA = (float*)malloc(M * K * RATIO * sizeof(float));

    float *matrixB = (float*)malloc(K * N * sizeof(float));
    int *rowPtrB = (int*)malloc((K + 1) * sizeof(int));
    int *colIndB = (int*)malloc(K * N * RATIO * sizeof(int));
    float *valB = (float*)malloc(K * N * RATIO * sizeof(float));

    prepareSparse(matrixA, M, K);
    prepareSparse(matrixB, K, N);

    // Show matrix
    showMatrix(matrixA, M, K);
    matrixToCSR(matrixA, M, K, rowPtrA, colIndA, valA);
    showCSR(M, K, rowPtrA, colIndA, valA);

    showMatrix(matrixB, K, N);
    matrixToCSR(matrixB, K, N, rowPtrB, colIndB, valB);
    showCSR(K, N, rowPtrB, colIndB, valB);

    return 0;
}