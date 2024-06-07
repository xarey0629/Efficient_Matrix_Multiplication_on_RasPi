#ifndef _BIN_H_
#define _BIN_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>

#define HASHING_CONST  2654435761

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
        assert(rows != 0);
        row_nnzflops = my_malloc<int>(rows);
        thread_row_offsets = my_malloc<int>(num_of_threads + 1);
        bin_size_leftShift_bits = my_malloc<char>(rows);
        local_hash_table_idx = my_malloc<int *>(num_of_threads);
        local_hash_table_val = my_malloc<double *>(num_of_threads);
        // Matrix C
        c_row_nnz = my_malloc<int>(rows);
    }


    long long int total_intprod;
    long long int max_intprod;
    int max_nnz;
    int num_of_threads;
    int min_hashTable_size;

    // Symbolic phase
    int *row_nnzflops;                  // Number of floating operations for each row
    int *thread_row_offsets;            // Indices of rows for each thread to start.
    char *bin_size_leftShift_bits;      // The number of bits to left shift the size of the hash table (We need only 1 byte to store). NOTE: 0 is saved for the empty row.
    int **local_hash_table_idx;         // Hash table for each thread: idx
    double **local_hash_table_val;      // Hash table for each thread: val
    
    // Output of Matrix C
    int *c_row_nnz;                     // Number of non-zero elements for each row in matrix C

    // Functions
    void set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols);
    void allocate_hash_tables(const int cols);
    void set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows);
    void set_thread_row_offsets(const int rows);
    void set_bin_leftShift_bits(const int rows, const int cols, const int min_ht_size);
};

// Compute the number of floating operations for each row
// Then compute the total number of floating operations
inline void BIN::set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows)
{
#pragma omp parallel
{
    int each_inter_prod = 0;
#pragma omp for
    for(int i = 0; i < rows; i++)
    {
        int nflops_per_row = 0;
        for(int j = arpt[i]; j < arpt[i + 1]; j++)
        {
            int col = acol[j];
            nflops_per_row += (brpt[col + 1] - brpt[col]);
        }
        row_nnzflops[i] = nflops_per_row;
        each_inter_prod += nflops_per_row;
    }
#pragma omp atomic
    total_intprod += each_inter_prod;
}
}

// Get prefix sum (could be parallelized)
void generateSequentialPrefixSum(int *in, int *out, int size)
{
    out[0] = 0;
    for(int i = 1; i < size; i++){
        out[i] = out[i - 1] + in[i - 1];
    }
} 

// Get prefix sum of row_nnzflops and the average number of non-zero elements per thread
// Then distribute equal work to each thread
inline void BIN::set_thread_row_offsets(const int rows)
{
    // Get prefix sum of row_nnzflops
    int *row_nnzflops_prefix_sum = my_malloc<int>(rows + 1);
    generateSequentialPrefixSum(row_nnzflops, row_nnzflops_prefix_sum, rows + 1);
    
    // Get the ceiling of average number of non-zero elements per thread
    int avg_nnz_per_thread = (total_intprod + num_of_threads - 1) / num_of_threads;

    // Set start row for each thread in order to distribute equal work to each thread.
    thread_row_offsets[0] = 0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int end = std::lower_bound(row_nnzflops_prefix_sum, row_nnzflops_prefix_sum + rows + 1, avg_nnz_per_thread * (tid + 1)) - row_nnzflops_prefix_sum;
        thread_row_offsets[tid + 1] = end;
    }
    thread_row_offsets[num_of_threads] = rows;
    
    my_free(row_nnzflops_prefix_sum);
}

// Compute how many entries in the hash table for each row before the multiplication
// The size of hash table is the power of 2. NOTE: The number of elements could be similar to the size of the hash table (lots of collisions).
inline void BIN::set_bin_leftShift_bits(const int rows, const int cols, const int min_ht_size)
{
    #pragma omp parallel for
    for(int i = 0; i < rows; i++)
    {
        int actual_size = std::min(row_nnzflops[i], cols);
        if(actual_size == 0)
            bin_size_leftShift_bits[i] = 0;
        else
        {
            int j = 0;
            while(actual_size > (min_ht_size << j)){
                j++;
            }
            bin_size_leftShift_bits[i] = j + 1; // 0 is saved for the empty row
        }
    }    
}


// Grouping and preparing hash table based on the number of floating operations
inline void BIN::set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols)
{
    set_intprod_num(arpt, acol, brpt, rows);
    set_thread_row_offsets(rows);
    set_bin_leftShift_bits(rows, cols, min_hashTable_size);
}

// Allocate hash table for each thread
inline void BIN::allocate_hash_tables(const int cols)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int ht_size = min_hashTable_size;
        for(int i = thread_row_offsets[tid]; i < thread_row_offsets[tid + 1]; i++)
        {
            if(bin_size_leftShift_bits[i] != 0)
            {
                ht_size = std::max(ht_size, min_hashTable_size << (bin_size_leftShift_bits[i] - 1));
            }
        }
        local_hash_table_idx[tid] = my_malloc<int>(ht_size);
        local_hash_table_val[tid] = my_malloc<double>(ht_size);
    }    
}

inline void hash_symbolic_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN &bin)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *map = bin.local_hash_table_idx[tid];
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int nz = 0;
            int left_shift = bin.bin_size_leftShift_bits[i]; // row by row
            if(left_shift > 0){
                int ht_size = bin.min_hashTable_size << (left_shift - 1);
                // Initialize hash table
                for(int j = 0; j < ht_size; j++)
                {
                    map[j] = -1;
                }

                // Fill hash table row by row.
                for(int j = arpt[i]; j < arpt[i + 1]; j++)
                {
                    int aCol = acol[j];
                    for(int k = brpt[aCol]; k < brpt[aCol + 1]; k++)
                    {
                        int bCol = bcol[k];
                        int hashKey = (bCol * HASHING_CONST) & (ht_size - 1);
                        // Linear probing
                        while(1) 
                        {
                            if(map[hashKey] == -1)
                            {
                                map[hashKey] = bCol;
                                nz++;
                                break;
                            }
                            else if(map[hashKey] == bCol)
                            {
                                break;
                            }
                            else
                            {
                                hashKey = (hashKey + 1) & (ht_size - 1);
                            }
                        }
                    }
                }
            }
            // Check the nnz;
            printf("The number of non-zero flops for row %d: is match? Ans: %B\n", i, bin.row_nnzflops[i] == nz ? true : false);
            bin.c_row_nnz[i] = nz;              
        }
    }
}

// Symbolic phase: Compute the number of non-zero elements for each row in matrix C
inline void hash_symbolic(const int *arpt, const int *acol, const int *brpt, const int *bcol, int *crpt, BIN &bin, const int nrow, int *c_nnz){
    hash_symbolic_kernel(arpt, acol, brpt, bcol, bin);
    generateSequentialPrefixSum(bin.c_row_nnz, crpt, nrow + 1);
    *c_nnz = crpt[nrow];
}


#endif // _BIN_H_