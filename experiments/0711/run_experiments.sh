#!/bin/bash

# Array of matrix sizes, doubling from 32 to 512
sizes=(32 64 128 256 512 1024 2048 4096)

# Array of densities (0.10 and 0.25)
densities=(0.10 0.25)

# Output file
output_file="perf_data.txt"

# Header for the output file
# echo "Program Matrix_Size Density Cycles Cache_References Cache_Misses L1_dcache_loads L1_dcache_load_misses LLC_loads LLC_load_misses" > $output_file

# Complie the programs
g++ -fopenmp -o simple_mul simple_mul.cpp && g++ -fopenmp -o single_hashing_SpGEMM single_hashing_SpGEMM.cpp && g++ -fopenmp -o multi_hashing_SpGEMM multi_hashing_SpGEMM.cpp && g++ -fopenmp -o multi_spArr_SpGEMM multi_spArr_SpGEMM.cpp

# Loop through matrix sizes
for density in "${densities[@]}"
do
    for size in "${sizes[@]}"
    # Loop through densities
    do
        # Execute perf stat for each program with size and density
        echo "Executing perf stat for density $density size $size..."

        # Execute simple_mul O(n^3)
        # echo "O(n^3) $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./simple_mul $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute single_SpGEMM
        # echo "Single_Hashing_SpGEMM $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./single_hashing_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute multi_SpGEMM
        echo "Multi_Hashing_SpGEMM $size $density " >> $output_file
        perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./multi_hashing_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute multi_spArr_SpGEMM
        echo "Multi_spArr_SpGEMM $size $density " >> $output_file
        perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./multi_spArr_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        echo "" >> $output_file
        echo "Done for density $density size $size"
    done
done

echo "Performance data recorded in $output_file"
