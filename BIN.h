#ifndef _BIN_H_
#define _BIN_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

template <typename T>
inline T *my_malloc(int arr_size)
{
    T *p = (T *)malloc(sizeof(T) * arr_size);
    if (p == NULL)
    {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

template <typename T>
inline void my_free(T *p)
{
    free(p);
}

struct BIN {
    BIN(int rows, int ht_size): total_intprod(0), max_intprod(0), max_nnz(0), num_of_threads(omp_get_max_threads()), min_hashTable_size(ht_size)
    {
        // assert(rows != 0);
        row_nnz = my_malloc<int>(rows);
        row_offset = my_malloc<int>(num_of_threads + 1);
        bin_id = my_malloc<char>(rows);
        local_hash_table_id = my_malloc<int *>(num_of_threads);
        local_hash_table_val = my_malloc<double *>(num_of_threads);
    }


    long long int total_intprod;
    long long int max_intprod;
    int max_nnz;
    int num_of_threads;
    int min_hashTable_size;

    // For output matirx C
    int *row_nnz;
    int *row_offset;
    char *bin_id;
    int **local_hash_table_id;
    double **local_hash_table_val;

    // Functions
    void set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols);
    void set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows);
    void set_rows_offset(const int rows);
    void set_bin_id(const int rows, const int cols, const int min_ht_size);
};

inline void BIN::set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows)
{
#pragma omp parallel
{
    int each_inter_prod = 0;
#pragma omp for
    for(int i = 0; i < rows; i++)
    {
        int nnz_per_row = 0;
        for(int j = arpt[i]; j < arpt[i + 1]; j++)
        {
            int col = acol[j];
            nnz_per_row += (brpt[col + 1] - brpt[col]);
        }
        row_nnz[i] = nnz_per_row;
        each_inter_prod += nnz_per_row;
    }
#pragma omp atomic
    total_intprod += each_inter_prod;
}
}

// Get prefix sum (could be parallelized)
void get_seq_prefix_sum(int *arr, int *prefix_sum, int size)
{
    prefix_sum[0] = 0;
    for(int i = 1; i < size; i++){
        prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1];
    }
} 

// Get prefix sum of row_nnz and the average number of non-zero elements per thread
// Then distribute equal work to each thread
inline void BIN::set_rows_offset(const int rows)
{
    // Get prefix sum of row_nnz
    int *row_prefix_sum = my_malloc<int>(rows + 1);
    get_seq_prefix_sum(row_nnz, row_prefix_sum, rows + 1);
    
    // Get the ceiling of average number of non-zero elements per thread
    int avg_nnz_per_thread = (total_intprod + num_of_threads - 1) / num_of_threads;

    // Set start and end points in order to distribute equal work to each thread
    row_offset[0] = 0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int end = std::lower_bound(row_prefix_sum, row_prefix_sum + rows + 1, avg_nnz_per_thread * (tid + 1)) - row_prefix_sum;
        row_offset[tid + 1] = end;
    }
    row_offset[num_of_threads] = rows;
}

// Compute how many entries in the hash table for each row before the multiplication
// Each size is the power of 2
inline void BIN::set_bin_id(const int rows, const int cols, const int min_ht_size)
{
    #pragma omp parallel for
    {
        for(int i = 0; i < rows; i++)
        {
            int actual_size = std::min(row_nnz[i], cols);
            if(actual_size == 0)
            {
                bin_id[i] = 0;
            }
            else
            {
                int j = 0;
                while(actual_size > (min_ht_size << j)){
                    j++;
                }
                bin_id[i] = j + 1; // 0 is saved for the empty row
            }
        }    
    }
}


/* grouping and preparing hash table based on the number of floating operations */
inline void BIN::set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols)
{
    set_intprod_num(arpt, acol, brpt, rows);
    set_rows_offset(rows);
    set_bin_id(rows, cols, min_hashTable_size);
}

#endif // _BIN_H_