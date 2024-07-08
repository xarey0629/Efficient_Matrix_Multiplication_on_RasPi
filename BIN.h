#ifndef _BIN_H_
#define _BIN_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>
#include <vector>
using namespace std;

#define HASHING_CONST           2654435761
#define MIN_HASH_TABLE_SIZE     8

// Memory allocation and free
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
// ----------------------------

// my BIN class
struct BIN {
    BIN(int rows, int ht_size = 8): total_intprod(0), max_intprod(0), max_nnz(0), num_of_threads(omp_get_max_threads()), min_hashTable_size(ht_size)
    {
        assert(rows != 0);
        // Symbolic phase
        row_nnzflops =              my_malloc<int>(rows);
        thread_row_offsets =        my_malloc<int>(num_of_threads + 1);
        
        // Hash table for each thread
        bin_size_leftShift_bits =   my_malloc<char>(rows);
        local_hash_table_idx =      my_malloc<int *>(num_of_threads);
        local_hash_table_val =      my_malloc<double *>(num_of_threads);
        
        // Matrix C
        c_row_nnz = my_malloc<int>(rows);
    }
    ~BIN()
    {
        my_free(row_nnzflops);
        my_free(thread_row_offsets);
        my_free(bin_size_leftShift_bits);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            my_free(local_hash_table_idx[tid]);
            my_free(local_hash_table_val[tid]);
        }
        my_free(local_hash_table_idx);
        my_free(local_hash_table_val);
        my_free(c_row_nnz);
    }

    // Variables
    long long int total_intprod;        // Total number of floating operations
    long long int max_intprod;          // Maximum number of floating operations
    int max_nnz;                        // Maximum number of non-zero elements in matrix C
    int num_of_threads;
    int min_hashTable_size;             // Default minimum size of hash table

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

/*  
    *************************** Functions start here ***************************
*/

/* 
Set intermediate product number for each row
* 1. Compute the number of floating operations of each row
* 2. Then compute the total number of floating operations
*/
inline void BIN::set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows)
{
#pragma omp parallel
{
    int each_inter_prod = 0;    // Number of floating operations for each thread
#pragma omp for
    for(int i = 0; i < rows; i++)
    {
        int nflops_per_row = 0;
        for(int j = arpt[i]; j < arpt[i + 1]; j++)
        {
            int a_col_idx = acol[j];
            nflops_per_row += (brpt[a_col_idx + 1] - brpt[a_col_idx]);
        }
        row_nnzflops[i] = nflops_per_row;
        each_inter_prod += nflops_per_row;
    }
#pragma omp atomic
    this->total_intprod += each_inter_prod;
}
}

// Set prefix sum of in into out. (could be parallelized)
void generateSequentialPrefixSum(int *in, int *out, int size)
{
    out[0] = 0;
    for(int i = 1; i < size; i++){
        out[i] = out[i - 1] + in[i - 1];
    }
} 


/* 
Set the start row for each thread in order to distribute equal work to each thread.
* 1. Get prefix sum of row_nnzflops
* 2. Get the average number of non-zero elements per thread
* 3. Distribute equal workload to each thread
*/
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
        int end = std::lower_bound(row_nnzflops_prefix_sum, row_nnzflops_prefix_sum + rows, avg_nnz_per_thread * (tid + 1)) - row_nnzflops_prefix_sum;
        thread_row_offsets[tid + 1] = end;
    }
    thread_row_offsets[num_of_threads] = rows;
    
    my_free(row_nnzflops_prefix_sum);
}

/*
Set the number of bits to left shift the size of the hash table
* 1. Compute how many entries in the hash table for each row before the multiplication
* 2. The size of hash table is the power of 2. 
NOTE: The number of elements could be similar to the size of the hash table (lots of collisions).
*/
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
            bin_size_leftShift_bits[i] = j + 1; // 0 is saved for the empty row, minus 1 when retrieving the size.
        }
    }    
}

/*
Grouping and preparing hash table based on the number of floating operations
* 1. Set the number of floating operations for each row
* 2. Set the start row for each thread in order to distribute equal work to each thread.
* 3. Set the number of bits to left shift the size of the hash table
*/
inline void BIN::set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols)
{
    set_intprod_num(arpt, acol, brpt, rows);
    set_thread_row_offsets(rows);
    set_bin_leftShift_bits(rows, cols, min_hashTable_size);
}

/*
Allocate hash table for each thread
* 1. Get the size of hash table for each row
* 2. Allocate hash table for each thread
*/
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
        // printf("Thread %d: Allocate hash table with size %d\n", tid, ht_size);
    }    
}

/* 
Sort the hash table by column indices and store the values back to CSR format
* 1. Create a pair vector to copy the column indices and values from the hash table
* 2. Sort the pair vector by column indices
* 3. Store the values back to CSR format
NOTE: The reason why we need to sort the hash table by column indices is that the hash table is filled out in a random order, but CSR needs increasing order.
*/
inline void sort_and_storeToCSR(int *map, double *map_val, int *ccol_start, float *cval_start, const int ht_size, const int vec_size, const bool SORT = true)
{
    int cnt = 0;
    if(SORT){
        // *** TODO ***: Rebuild hash table with a pair array. But the time sorting -1 indices should be considered.
        std::vector<std::pair<int, double>> vec;
        for(int i = 0; i < ht_size; i++)
        {
            if(map[i] != -1) vec.push_back(std::make_pair(map[i], map_val[i]));
        }
        assert(vec.size() == vec_size);

        // Sort the hash table by column indices and store the values back to CSR format
        sort(vec.begin(), vec.end());
        for(int i = 0; i < vec.size(); i++)
        {
            ccol_start[i] = vec[i].first;
            cval_start[i] = vec[i].second;
        }
    }
    else{ // No sorting, if unnecessary
        for(int i = 0; i < ht_size; i++)
        {
            if(map[i] != -1)
            {
                ccol_start[cnt] = map[i];
                cval_start[cnt] = map_val[i];
                cnt++;
            }
        }
    }
}

/*
Hashing SpGEMM symbolic kernel --> Compute the number of non-zero elements for each row in matrix C
* 1. Initialize hash table for each thread
* 2. Fill out hash table row by row
* 3. Compute the number of non-zero elements for each row in matrix C
*/
inline void hash_symbolic_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN &bin)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *map = bin.local_hash_table_idx[tid];
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int nz = 0;
            int ht_size_left_shift_bits = bin.bin_size_leftShift_bits[i]; // row by row
            if(ht_size_left_shift_bits != 0){
                int ht_size = bin.min_hashTable_size << (ht_size_left_shift_bits - 1);
                // Initialize hash table
                for(int j = 0; j < ht_size; j++)
                {
                    map[j] = -1;
                }

                // Fill out hash table row by row.
                for(int j = arpt[i]; j < arpt[i + 1]; j++)
                {
                    int aCol = acol[j];
                    for(int k = brpt[aCol]; k < brpt[aCol + 1]; k++)
                    {
                        int bCol = bcol[k]; // Use column index of matrix B as the key.
                        int hashKey = (bCol * HASHING_CONST) & (ht_size - 1);
                        
                        // Linear probing
                        while(1) 
                        {
                            if(map[hashKey] == -1) // The column index doesn't exist, insert it and increment the number of non-zero elements in the current row of matrix C.
                            {
                                map[hashKey] = bCol;
                                nz++;
                                break;
                            }
                            else if(map[hashKey] == bCol) break; // If the same column index is found, do nothing.
                            else hashKey = (hashKey + 1) & (ht_size - 1);
                        }
                    }
                }
                // printf("Thread %d finished row %d\n", tid, i);
            }
            bin.c_row_nnz[i] = nz; // Actual nnz of each row in matrix C              
        }
    }
}

/*
Hashing SpGEMM symbolic phase execution
* 1. Execute symbolic kernel
* 2. Prefix sum up number of non-zero elements for each row in matrix C
*/
inline void hash_symbolic(const int *arpt, const int *acol, const int *brpt, const int *bcol, int *crpt, BIN &bin, const int nrow, int *c_nnz){
    hash_symbolic_kernel(arpt, acol, brpt, bcol, bin);
    generateSequentialPrefixSum(bin.c_row_nnz, crpt, nrow + 1);
    *c_nnz = crpt[nrow];  // Set the total number of non-zero elements in matrix C
}

/* 
Hashing SpGEMM numeric phase execution
* 1. Compute the actual values of matrix C
* 2. Store the values back to CSR format
* 3. Method 1: If the matrix C is in normal format, just fill out the values.
* 4. Method 2: If the matrix C is in CSR format, we need to sort the hash table by column indices, then store values.
*/
inline void hash_numeric(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, int *crpt, int *ccol, float *cval, BIN &bin){
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *map = bin.local_hash_table_idx[tid];
        double *map_val = bin.local_hash_table_val[tid];

        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++) // In a row by row manner
        {
            int ht_size_left_shift_bits = bin.bin_size_leftShift_bits[i];
            if(ht_size_left_shift_bits > 0)
            {
                int idx_offset = crpt[i];
                int ht_size = bin.min_hashTable_size << (ht_size_left_shift_bits - 1);
                // Initialize hash table
                for(int j = 0; j < ht_size; j++)
                {
                    map[j] = -1;
                }
                // Fill out hash table row by row.
                for(int j = arpt[i]; j < arpt[i + 1]; j++)
                {
                    int aCol = acol[j];
                    for(int k = brpt[aCol]; k < brpt[aCol + 1]; k++)
                    {
                        int bCol = bcol[k]; // Use column index of matrix B as the key.
                        int hashKey = (bCol * HASHING_CONST) & (ht_size - 1);
                        // Linear probing
                        while(1) 
                        {
                            if(map[hashKey] == -1)
                            {
                                map[hashKey] = bCol;
                                map_val[hashKey] = aval[j] * bval[k];
                                break;
                            }
                            else if(map[hashKey] == bCol)
                            {
                                map_val[hashKey] += aval[j] * bval[k];
                                break;
                            }
                            else hashKey = (hashKey + 1) & (ht_size - 1);
                        }
                    }
                }
                // Fill out the actual values of matrix C
                // Method 1: If the matrix C is in normal format, just fill out the values.    
                // Method 2: If the matrix C is in CSR format, we need to sort the hash table by column indices, then store values.
                sort_and_storeToCSR(map, map_val, ccol + idx_offset, cval + idx_offset, ht_size, bin.c_row_nnz[i]);
            }
        }
    }
}

/*
Main function to execute hashing SpGEMM execution
* 1. Initialize BIN object
* 2. Symbolic phase
* 3. Numeric phase
*/
inline void execute_hashing_SpGEMM(const int *arpt, const int *acol, const float *aval, 
                                        const int *brpt, const int *bcol, const float *bval, 
                                        int *&crpt, int *&ccol, float *&cval, const int nrow, const int ncol)
{
    // Initialize BIN object
    BIN myBin(nrow, MIN_HASH_TABLE_SIZE);               // Create a BIN object.
    myBin.set_max_bin(arpt, acol, brpt, nrow, ncol);    // Load balancing and set the size of hash table, which is flops(row_i), for each row.
    myBin.allocate_hash_tables(ncol);                   // Allocate hash table for each thread.

    // Symbolic phase
    int c_nnz = 0;                                                          // nnz(C), dereferenced by hash_symbolic.
    crpt = my_malloc<int>(nrow + 1);
    hash_symbolic(arpt, acol, brpt, bcol, crpt, myBin, nrow, &c_nnz);       // Symbolic phase, and get nnz(C).
    ccol = my_malloc<int>(c_nnz);
    cval = my_malloc<float>(c_nnz);

    // // print each row's nnz
    // for(int i = 0; i < nrow; i++){
    //     printf("row_nnzflops[%d]: %d\n", i, myBin.row_nnzflops[i]);
    // }
    // // print thread_row_offsets
    // for(int i = 0; i < myBin.num_of_threads + 1; i++){
    //     printf("thread_row_offsets[%d]: %d\n", i, myBin.thread_row_offsets[i]);
    // }

    // Numeric phase
    hash_numeric(arpt, acol, aval, brpt, bcol, bval, crpt, ccol, cval, myBin);
}

/*  
    *************************** Functions end here ***************************
*/


#endif // _BIN_H_