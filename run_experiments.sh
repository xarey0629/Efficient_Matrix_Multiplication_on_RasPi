#!/bin/bash

# List of matrix sizes
# sizes=(32 64 128 256 512 1024 2048 4096)
sizes=(32 64 128 256)


# Output file
output_file="perf_results.txt"

# Write header to the output file
echo "Matrix Size, Function, Cycles, Cache References, Cache Misses, L1 Dcache Loads, L1 Dcache Load Misses, LLC Loads, LLC Load Misses" > $output_file

# Loop over each matrix size
for size in "${sizes[@]}"; do
    echo "Running experiment with matrix size: $size"

    # Run perf for each specified function
    for func in simpleMatrixMultiplication single_threaded_hashing_SpGEMM multi_threaded_hashing_SpGEMM; do
        echo "Profiling function: $func"

        # Record performance data using perf
        perf record -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses -g -F 40000 -o perf_data_${size}_${func}.data ./SpGEMM $size

        # Generate performance report and filter for the specific function
        perf report -i perf_data_${size}_${func}.data --stdio --no-children --symbol-filter=$func | \
        awk -v size=$size -v func=$func 'BEGIN {FS=","; OFS=","} /cycles/ {cycles=$1} /cache-references/ {cache_references=$1} /cache-misses/ {cache_misses=$1} /L1-dcache-loads/ {l1_dcache_loads=$1} /L1-dcache-load-misses/ {l1_dcache_load_misses=$1} /LLC-loads/ {llc_loads=$1} /LLC-load-misses/ {llc_load_misses=$1} END {print size, func, cycles, cache_references, cache_misses, l1_dcache_loads, l1_dcache_load_misses, llc_loads, llc_load_misses}' >> $output_file
    done
done
