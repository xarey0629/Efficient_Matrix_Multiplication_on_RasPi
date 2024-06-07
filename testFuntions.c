#ifndef _test_h_
#define _test_h_

void test_rows_density(int M, double *matrixA, double *matrixB){
    for(int row = 0; row < M; row++){
        int cnt = 0;
        for(int i = 0; i < M; i++){
            for(int j = 0; j < M; j++){
                if(matrixA[row * M + i] != 0 && matrixB[i * M + j] != 0){
                    cnt++;
                }    
            }
        }
        printf("row %d: %d\n", row, cnt);
    }
}

#endif // _test_h_