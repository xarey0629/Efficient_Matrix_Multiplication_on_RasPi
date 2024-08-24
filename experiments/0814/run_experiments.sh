#!/bin/bash

# Array of matrix sizes, doubling from 32 to 8192
# sizes=(32 64 128 256 512 1024 2048 4096 8192)
sizes=(32 64 128 256 512 1024 2048 4096 8192)


# Array of densities (0.001 to 0.01)
densities=(0.001 0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.020)
# densities=(0.01 0.025 0.05 0.075 0.1 0.125)

# Complie the programs
g++ -fopenmp -o SpGEMM SpGEMM.cpp 

# Remove the output file
rm -f indbuf_0814.txt

# Loop through matrix sizes
for size in "${sizes[@]}"
do
    for density in "${densities[@]}"
    # Loop through densities
    do
        # Execute perf stat for each program with size and density
        echo "Executing, density: $density, size: $size..."

        # Execute three times
        for i in {1..3}
        do
            ./SpGEMM $size $density
        done

        echo "Done for density $density size $size"
    done
done