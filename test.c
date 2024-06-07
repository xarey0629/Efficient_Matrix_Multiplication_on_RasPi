#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv){
    #pragma omp parallel
    {
        printf("Hello from process: %d, total: %d\n", omp_get_thread_num(), omp_get_max_threads());
    }
    return 0;
}