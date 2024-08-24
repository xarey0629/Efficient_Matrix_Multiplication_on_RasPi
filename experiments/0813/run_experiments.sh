#!/bin/bash

# Array of matrix sizes, doubling from 32 to 8192
sizes=(32 64 128 256 512 1024 2048 4096 8192)

# Array of densities (0.001 to 0.01)
# densities=(0.001 0.002 0.004 0.006 0.008 0.01)
densities=(0.01 0.025 0.05 0.075 0.1 0.125)


# Current date
curr_date="0813"

# Output file
rm -f perf_data_$curr_date.txt
output_file="perf_data_$curr_date.txt"

# Header for the output file
# echo "Program Matrix_Size Density Cycles Cache_References Cache_Misses L1_dcache_loads L1_dcache_load_misses LLC_loads LLC_load_misses" > $output_file

# Complie the programs
g++ -fopenmp -o hashing_SpGEMM hashing_SpGEMM.cpp && g++ -fopenmp -o spArr_SpGEMM spArr_SpGEMM.cpp

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
        # echo "Multi_Hashing_SpGEMM $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./multi_hashing_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute multi_spArr_SpGEMM
        # echo "Multi_spArr_SpGEMM $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./multi_spArr_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute hashing_woN_SpGEMM
        echo "hashing_SpGEMM $size $density " >> $output_file
        perf stat -e cycles,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./hashing_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # # Execute hashing_wN_SpGEMM
        # echo "hashing_SpGEMM_NEON $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./hashing_wN_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # Execute spArr_woN_SpGEMM
        echo "spArr_SpGEMM $size $density " >> $output_file
        perf stat -e cycles,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./spArr_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file

        # # Execute spArr_wN_SpGEMM
        # echo "spArr_SpGEMM_NEON $size $density " >> $output_file
        # perf stat -e cycles,cache-references,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ./spArr_wN_SpGEMM $size $density 2>&1 | grep -E '^[ ]+[0-9]' >> $output_file
        
        echo "" >> $output_file
        echo "Done for density $density size $size"
    done
done

echo "Performance data recorded in $output_file"
