#ifndef _BIN_H_
#define _BIN_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>
#include <vector>
#include <arm_neon.h>

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
inline void my_free(T* p)
{
    if(p == nullptr) return;
    free(p);
}
// ----------------------------

// my BIN class
struct BIN {
    BIN(int rows, int cols, int num_of_threads): rows(rows), cols(cols), total_intprod(0), max_intprod(0), c_nnz(0), num_of_threads(num_of_threads), min_hashTable_size(MIN_HASH_TABLE_SIZE)
    {
        assert(rows != 0);
        // Symbolic and Load balancing
        row_nnzflops =              my_malloc<int>(rows);
        thread_row_offsets =        my_malloc<int>(num_of_threads + 1);
        
        // Matrix C
        c_row_nnz = my_malloc<int>(rows);
    }
    ~BIN()
    {
        my_free(row_nnzflops);
        my_free(thread_row_offsets);
        my_free(bin_size_leftShift_bits);
        if(local_hash_table_idx != nullptr)
        {
            #pragma omp parallel num_threads(num_of_threads)
            {
                int tid = omp_get_thread_num();
                my_free(local_hash_table_idx[tid]);
                my_free(local_hash_table_val[tid]);
            }
        }
        my_free(local_hash_table_idx);
        my_free(local_hash_table_val);
        my_free(c_row_nnz);
    }

    // Matrix Info
    int rows;                           // Number of rows
    int cols;                           // Number of columns

    // Variables
    long long int total_intprod;        // Total number of floating operations
    long long int max_intprod;          // Maximum number of floating operations
    int c_nnz;                          // Maximum number of non-zero elements in matrix C
    int num_of_threads;
    int min_hashTable_size;             // Default minimum size of hash table

    // Load balancing
    int *row_nnzflops;                  // Number of floating operations for each row
    int *thread_row_offsets;            // Indices of rows for each thread to start.
    // Symbolic phase
    char *bin_size_leftShift_bits = nullptr;      // The number of bits to left shift the size of the hash table (We need only 1 byte to store). NOTE: 0 is saved for the empty row.
    int **local_hash_table_idx = nullptr;         // Hash table for each thread: idx
    float **local_hash_table_val = nullptr;       // Hash table for each thread: val
    // SpArr
    vector<vector<double>> local_spArr_mat;     // A 2D-vector to store the values of matrix C
    vector<vector<int>> local_spArr_indBuf;     // A 2D-vector to store the column indices of matrix C
    
    // Output of Matrix C
    int *c_row_nnz;                     // Number of non-zero elements for each row in matrix C

    // Functions
    void set_max_bin(const int *arpt, const int *acol, const int *brpt, const int rows, const int cols);
    void set_intprod_num(const int *arpt, const int *acol, const int *brpt, const int rows);
    void set_thread_row_offsets(const int rows);
    /* Hashing */
    void set_bin_leftShift_bits();
    void allocate_hash_tables();
    /* SpArr */
    void allocate_spArrs();
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
#pragma omp parallel num_threads(num_of_threads)
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


/* Load balancer
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
    #pragma omp parallel num_threads(num_of_threads)
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
inline void BIN::set_bin_leftShift_bits()
{
    bin_size_leftShift_bits = my_malloc<char>(this->rows);
    #pragma omp parallel for num_threads(num_of_threads)
    for(int i = 0; i < rows; i++)
    {
        int actual_size = std::min(row_nnzflops[i], cols);
        if(actual_size == 0)
            bin_size_leftShift_bits[i] = 0;
        else
        {
            int j = 0;
            while(actual_size > (min_hashTable_size << j)){
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
    set_bin_leftShift_bits();
}

/*
Allocate hash table for each thread
* 1. Get the size of hash table for each row
* 2. Allocate hash table for each thread
*/
inline void BIN::allocate_hash_tables()
{
    // Hash table for each thread
    local_hash_table_idx = my_malloc<int *>(num_of_threads);
    local_hash_table_val = my_malloc<float *>(num_of_threads);

    #pragma omp parallel num_threads(num_of_threads)
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
        local_hash_table_val[tid] = my_malloc<float>(ht_size);
        // printf("Thread %d: Allocate hash table with size %d\n", tid, ht_size);
    }    
}

/* Resize the size of rows (num of columns) in local_spArr_mat */
inline void BIN::allocate_spArrs(){
    local_spArr_mat.resize(rows);
    for(int i = 0; i < local_spArr_mat.size(); i++){
        local_spArr_mat[i].resize(cols, 0);
    }
    local_spArr_indBuf.resize(rows);
}

/* 
Sort the hash table by column indices and store the values back to CSR format
* 1. Create a pair vector to copy the column indices and values from the hash table
* 2. Sort the pair vector by column indices
* 3. Store the values back to CSR format
NOTE: The reason why we need to sort the hash table by column indices is that the hash table is filled out in a random order, but CSR needs increasing order.
*/
inline void sort_and_storeToCSR(int *map, float *map_val, int *ccol_start, float *cval_start, const int ht_size, const int vec_size, const bool SORT = true)
{
    int cnt = 0;
    if(SORT){
        // *** TODO ***: Rebuild hash table with a pair array. But the time sorting -1 indices should be considered.
        std::vector<std::pair<int, float>> vec;
        for(int i = 0; i < ht_size; i++)
        {
            if(map[i] != -1) vec.push_back(std::make_pair(map[i], map_val[i]));
        }
        // assert(vec.size() == vec_size);

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
    #pragma omp parallel num_threads(bin.num_of_threads)
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

// Extract most significant 2 bits of a 32-bit integer of each element in a vector into a 32 bit integer
uint32_t extractMSB2(const uint32x4_t &v)
{
    uint32_t res = 0;
    res |= (vgetq_lane_u32(v, 0) & 0xFF000000) >> 24;
    res |= (vgetq_lane_u32(v, 1) & 0xFF000000) >> 16;
    res |= (vgetq_lane_u32(v, 2) & 0xFF000000) >> 8;
    res |= (vgetq_lane_u32(v, 3) & 0xFF000000);
    return res;
}

/*
NEON_hashing_SpGEMM_kernel
*/
#define VEC_LENGTH      4   // NEON uses 128-bit vectors which can hold 4 32-bit integers
#define VEC_LENGTH_BIT  2   // log2(VEC_LENGTH) = 2
#define MIN_HT_S        16  // Minimum hash table size (can be adjusted as needed)
inline void hash_symbolic_vec_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN &bin)
{
    const int32x4_t init_m = vdupq_n_s32(-1);
    const uint32x4_t true_m = vdupq_n_u32(0xffffffff);

#pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num(); 
        int *check = bin.local_hash_table_idx[tid];
        for (int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++) {
            int32x4_t   key_m, check_m;
            uint32x4_t  mask_m;
            int32_t    mask;

            int nz = 0;
            int ht_size_left_shift_bits = bin.bin_size_leftShift_bits[i];   // row by row
            if(ht_size_left_shift_bits != 0){
                int table_size = bin.min_hashTable_size << (ht_size_left_shift_bits - 1);
                int ht_size = table_size >> VEC_LENGTH_BIT;                 // the number of chunks (1 chunk = VEC_LENGTH elements)
                // Initialize hash table
                for(int j = 0; j < table_size; j++)
                {
                    check[j] = -1;
                }

                // Fill out hash table row by row.
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) 
                {
                    int t_acol = acol[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) 
                    {
                        int key = bcol[k];
                        int hashKey = (key * HASHING_CONST) & (ht_size - 1);
                        key_m = vdupq_n_s32(key);

                        // Loop for hash probing
                        while (1) 
                        {
                            // Check whether the key is in hash table.
                            check_m = vld1q_s32(check + (hashKey << VEC_LENGTH_BIT));
                            mask_m = vceqq_s32(key_m, check_m);
                            mask = extractMSB2(mask_m);
                            if (mask != 0) {
                                break;
                            } else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
                                mask_m = vceqq_s32(check_m, init_m);
                                mask = extractMSB2(mask_m);
                                cur_nz = __builtin_popcount(~mask & 0xffffffff) >> 8;

                                if (cur_nz < VEC_LENGTH) { // if it is not filled, push the entry to the table
                                    check[(hashKey << VEC_LENGTH_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                } else { // if is filled, check next chunk (linear probing)
                                    hashKey = (hashKey + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.c_row_nnz[i] = nz;
        }
    }
}

/*
Hashing SpGEMM symbolic phase execution
* 1. Execute symbolic kernel
* 2. Prefix sum up number of non-zero elements for each row in matrix C
*/
inline void hash_symbolic(const int *arpt, const int *acol, const int *brpt, const int *bcol, int *crpt, BIN &bin, const int nrow, bool NEON = true){
    if(NEON)    
        hash_symbolic_vec_kernel(arpt, acol, brpt, bcol, bin);
    else        
        hash_symbolic_kernel(arpt, acol, brpt, bcol, bin);
    generateSequentialPrefixSum(bin.c_row_nnz, crpt, nrow + 1);
    // *c_nnz = crpt[nrow];  // Set the total number of non-zero elements in matrix C
    bin.c_nnz = crpt[nrow]; // Set the maximum number of non-zero elements in matrix C
}

/* 
Hashing SpGEMM numeric phase execution
* 1. Compute the actual values of matrix C
* 2. Store the values back to CSR format
* 3. Method 1: If the matrix C is in normal format, just fill out the values.
* 4. Method 2: If the matrix C is in CSR format, we need to sort the hash table by column indices, then store values.
*/
inline void hash_numeric_kernel(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, int *crpt, int *ccol, float *cval, BIN &bin){
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        int *map = bin.local_hash_table_idx[tid];
        float *map_val = bin.local_hash_table_val[tid];

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
    // TODO: Avoid the second hashing.
    // May try to allocate memory for matrix C here.
    // and then sort_and_storeToCSR().
}


/*
NEON_hashing_SpGEMM_kernel
*/
inline void hash_numeric_vec_kernel(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, int *crpt, int *ccol, float *cval, BIN &bin)
{
    const int32x4_t init_m = vdupq_n_s32(-1);
    const uint32x4_t true_m = vdupq_n_u32(0xffffffff);

#pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num(); 
        int *check = bin.local_hash_table_idx[tid];
        float *map_val = bin.local_hash_table_val[tid];

        for (int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++) {
            int32x4_t   key_m, check_m;
            uint32x4_t  mask_m;
            int32_t    mask;

            int ht_size_left_shift_bits = bin.bin_size_leftShift_bits[i];   // row by row
            if(ht_size_left_shift_bits != 0){
                int idx_offset = crpt[i];
                int table_size = bin.min_hashTable_size << (ht_size_left_shift_bits - 1);
                int ht_size = table_size >> VEC_LENGTH_BIT;                 // the number of chunks (1 chunk = VEC_LENGTH elements)
                // Initialize hash table
                for(int j = 0; j < table_size; j++)
                {
                    check[j] = -1;
                }

                // Fill out hash table row by row.
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) 
                {
                    int t_acol = acol[j];
                    float t_aval = aval[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) 
                    {
                        float tmp_val = t_aval * bval[k];
                        int key = bcol[k];
                        int hashKey = (key * HASHING_CONST) & (ht_size - 1);
                        key_m = vdupq_n_s32(key);

                        // Loop for hash probing
                        while (1) 
                        {
                            // Check whether the key is in hash table.
                            check_m = vld1q_s32(check + (hashKey << VEC_LENGTH_BIT));
                            mask_m = vceqq_s32(key_m, check_m);
                            mask = extractMSB2(mask_m);                                     // First element will be stored in the first 8 bits of the mask.
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 3;                      // Count trailing zeros. 
                                map_val[(hashKey << VEC_LENGTH_BIT) + target] += tmp_val;
                                break;
                            } else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
                                mask_m = vceqq_s32(check_m, init_m);
                                mask = extractMSB2(mask_m);
                                cur_nz = __builtin_popcount(~mask & 0xffffffff) >> 3;

                                if (cur_nz < VEC_LENGTH) { // if it is not filled, push the entry to the table
                                    check[(hashKey << VEC_LENGTH_BIT) + cur_nz] = key;
                                    map_val[(hashKey << VEC_LENGTH_BIT) + cur_nz] = tmp_val;
                                    break;
                                } else { // if is filled, check next chunk (linear probing)
                                    hashKey = (hashKey + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_storeToCSR(check, map_val, ccol + idx_offset, cval + idx_offset, table_size, bin.c_row_nnz[i]);
            }
        }
    }
}

/* 
Hashing SpGEMM numeric phase execution 
*/
inline void hash_numeric(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, int *crpt, int *ccol, float *cval, BIN &bin, bool NEON = true){
    if(NEON)    
        hash_numeric_vec_kernel(arpt, acol, aval, brpt, bcol, bval, crpt, ccol, cval, bin);
    else        
        hash_numeric_kernel(arpt, acol, aval, brpt, bcol, bval, crpt, ccol, cval, bin);
}

/*
Main function to execute hashing SpGEMM execution
* 1. Initialize BIN object
* 2. Symbolic phase
* 3. Numeric phase
*/
inline void execute_hashing_SpGEMM(const int *arpt, const int *acol, const float *aval, 
                                        const int *brpt, const int *bcol, const float *bval, 
                                        int *&crpt, int *&ccol, float *&cval, const int nrow, const int ncol,
                                        bool NEON, int num_of_threads = omp_get_max_threads())
{
    // Initialize BIN object
    BIN myBin(nrow, ncol, num_of_threads);   // Create a BIN object.
    myBin.set_max_bin(arpt, acol, brpt, nrow, ncol);        // Load balancing and set the size of hash table, which is flops(row_i), for each row.
    myBin.allocate_hash_tables();                       // Allocate hash table for each thread.

    // Symbolic phase
    int c_nnz = 0;                                                                      // nnz(C), dereferenced by hash_symbolic.
    crpt = my_malloc<int>(nrow + 1);
    hash_symbolic(arpt, acol, brpt, bcol, crpt, myBin, nrow, NEON);   // Symbolic phase, and get nnz(C).
    ccol = my_malloc<int>(myBin.c_nnz);
    cval = my_malloc<float>(myBin.c_nnz);

    // // print each row's nnz
    // for(int i = 0; i < nrow; i++){
    //     printf("row_nnzflops[%d]: %d\n", i, myBin.row_nnzflops[i]);
    // }
    // // print thread_row_offsets
    // for(int i = 0; i < myBin.num_of_threads + 1; i++){
    //     printf("thread_row_offsets[%d]: %d\n", i, myBin.thread_row_offsets[i]);
    // }

    // Numeric phase
    hash_numeric(arpt, acol, aval, brpt, bcol, bval, crpt, ccol, cval, myBin, NEON);
}

/* spArr_SpGEMM kernel*/
inline void spArr_SpGEMM_kernel(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, BIN &bin)
{
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int row_nz = 0;
            for(int j = arpt[i]; j < arpt[i + 1]; j++)
            {
                int aCol = acol[j];
                for(int k = brpt[aCol]; k < brpt[aCol + 1]; k++)
                {
                    int bCol = bcol[k];
                    if(bin.local_spArr_mat[i][bCol] == 0){
                        row_nz++;
                    }
                    bin.local_spArr_mat[i][bCol] += aval[j] * bval[k];
                }
            }
            bin.c_row_nnz[i] = row_nz;
        }
    }
}

/* spArr_SpGEMM kernel with indBuf */
inline void spArr_SpGEMM_kernel_v2(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, BIN &bin)
{
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int row_nz = 0;
            for(int j = arpt[i]; j < arpt[i + 1]; j++)
            {
                int aCol = acol[j];
                for(int k = brpt[aCol]; k < brpt[aCol + 1]; k++)
                {
                    int bCol = bcol[k];
                    if(bin.local_spArr_mat[i][bCol] == 0){
                        bin.local_spArr_indBuf[i].push_back(bCol);
                        row_nz++;
                    }
                    bin.local_spArr_mat[i][bCol] += aval[j] * bval[k];
                }
            }
            bin.c_row_nnz[i] = row_nz;
        }
    }
}

/* NEON_spArr_SpGEMM kernel */
inline void NEON_SpArr_SpGEMM_kernel(const int *arpt, const int *acol, const float *aval, const int *brpt, const int *bcol, const float *bval, BIN &bin)
{
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int row_nz = 0;
            for(int j = arpt[i]; j < arpt[i + 1]; j++)
            {
                int aCol = acol[j];
                float32x4_t aVal = vdupq_n_f32(aval[j]);
                for(int k = brpt[aCol]; k < brpt[aCol + 1];)
                {
                    if(k + 4 <= brpt[aCol + 1])
                    {
                        int bCol[4] = {bcol[k], bcol[k + 1], bcol[k + 2], bcol[k + 3]};
                        float32x4_t bVal = vld1q_f32(&bval[k]);
                        float32x4_t inter_res = vmulq_f32(aVal, bVal);
                        for(int l = 0; l < 4; l++)
                        {
                            if(bin.local_spArr_mat[i][bCol[l]] == 0){
                                bin.local_spArr_indBuf[i].push_back(bCol[l]);
                                row_nz++;
                            }
                        }
                        bin.local_spArr_mat[i][bCol[0]] += vgetq_lane_f32(inter_res, 0);
                        bin.local_spArr_mat[i][bCol[1]] += vgetq_lane_f32(inter_res, 1);
                        bin.local_spArr_mat[i][bCol[2]] += vgetq_lane_f32(inter_res, 2);
                        bin.local_spArr_mat[i][bCol[3]] += vgetq_lane_f32(inter_res, 3);                        
                        k += 4;
                    }
                    else
                    {
                        int bCol = bcol[k];
                        if(bin.local_spArr_mat[i][bCol] == 0){
                            bin.local_spArr_indBuf[i].push_back(bCol);
                            row_nz++;
                        }
                        bin.local_spArr_mat[i][bCol] += aval[j] * bval[k];
                        k++;
                    }
                }
            }
            bin.c_row_nnz[i] = row_nz;
        }
    }
}

/* Scan matrix and store it into CST*/
inline void spArr_SpGEMM_store(const int *crpt, int *ccol, float *cval, BIN &bin)
{
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int idx_offset = crpt[i];
            int row_nz = bin.c_row_nnz[i];
            int cnt = 0;
            for(int j = 0; j < bin.local_spArr_mat[i].size(); j++)
            {
                if(bin.local_spArr_mat[i][j] != 0)
                {
                    ccol[idx_offset + cnt] = j;
                    cval[idx_offset + cnt] = bin.local_spArr_mat[i][j];
                    cnt++;
                }
            }
            // assert(cnt == row_nz);
            // printf("Counted Nnz of row %d: %d\n", i, cnt);
            // printf("Predicted Nnz of row %d: %d\n", i, bin.c_row_nnz[i]);
        }
    }
}

/* Scan index buffer and access to matrix, then store it into CST*/
inline void spArr_SpGEMM_store_v2(const int *crpt, int *ccol, float *cval, BIN &bin)
{
    #pragma omp parallel num_threads(bin.num_of_threads)
    {
        int tid = omp_get_thread_num();
        for(int i = bin.thread_row_offsets[tid]; i < bin.thread_row_offsets[tid + 1]; i++)
        {
            int idx_offset = crpt[i];
            int row_nz = bin.c_row_nnz[i];
            int cnt = 0;
            for(int j = 0; j < bin.local_spArr_indBuf[i].size(); j++)
            {
                int colInd = bin.local_spArr_indBuf[i][j];
                ccol[idx_offset + cnt] = colInd;
                cval[idx_offset + cnt] = bin.local_spArr_mat[i][colInd];
                cnt++;
            }
            // assert(cnt == row_nz);
            // printf("Counted Nnz of row %d: %d\n", i, cnt);
            // printf("Predicted Nnz of row %d: %d\n", i, bin.c_row_nnz[i]);
        }
    }
}

/*
Main function to execute spArr SpGEMM execution
* 1. Initialize BIN object
* 2. spArr kernel
* 3. Store
*/
inline void execute_spArr_SpGEMM(const int *arpt, const int *acol, const float *aval, 
                                        const int *brpt, const int *bcol, const float *bval, 
                                        int *&crpt, int *&ccol, float *&cval, const int nrow, const int ncol,
                                        bool NEON, bool indBuf, int num_of_threads = omp_get_max_threads())
{
    // Initialize BIN object
    BIN myBin(nrow, ncol, num_of_threads);   // Create a BIN object.
    myBin.set_intprod_num(arpt, acol, brpt, nrow);
    myBin.set_thread_row_offsets(nrow);
    myBin.allocate_spArrs();                            // Allocate a 2D-vector in size of rows x cols.

    // spArr kernel
    if(NEON) 
        NEON_SpArr_SpGEMM_kernel(arpt, acol, aval, brpt, bcol, bval, myBin);
    else{
        if(indBuf)
            spArr_SpGEMM_kernel_v2(arpt, acol, aval, brpt, bcol, bval, myBin);
        else
            spArr_SpGEMM_kernel(arpt, acol, aval, brpt, bcol, bval, myBin);
    }
    
    crpt = my_malloc<int>(nrow + 1);
    generateSequentialPrefixSum(myBin.c_row_nnz, crpt, nrow + 1);
    myBin.c_nnz = crpt[nrow];
    ccol = my_malloc<int>(crpt[nrow]);
    cval = my_malloc<float>(crpt[nrow]);

    // Store
    if(indBuf)
        spArr_SpGEMM_store_v2(crpt, ccol, cval, myBin);
    else
        spArr_SpGEMM_store(crpt, ccol, cval, myBin);
}

/*  
    *************************** Functions end here ***************************
*/


#endif // _BIN_H_