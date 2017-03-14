#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab3IO.h"
//****************************************************
// modified from the serial solution in serialTester 
//****************************************************
int main(int argc, char* argv[]){
	int i, j, k, size;
	double** Au;
	double* X;
	double temp, error, Xnorm;
	int* index;
	FILE* fp;

    //get the number of threads
    int thread_count = atoi(argv[1]);

	/*Load the datasize and verify it*/
	Lab3LoadInput(&Au, &size);
	/*Calculate the solution by serial code*/
	X = CreateVec(size);
    index = malloc(size * sizeof(int));


    # pragma omp parallel for num threads(thread_count)    
    for (i = 0; i < size; ++i){
        index[i] = i;
    }

    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
        for (k = 0; k < size - 1; ++k){
            /*Pivoting*/
            temp = 0;
            for (i = k, j = 0; i < size; ++i)
                if (temp < Au[index[i]][k] * Au[index[i]][k]){
                    temp = Au[index[i]][k] * Au[index[i]][k];
                    j = i;
                }
            if (j != k)/*swap*/{
                i = index[j];
                index[j] = index[k];
                index[k] = i;
            }
            /*calculating*/
            for (i = k + 1; i < size; ++i){
                temp = Au[index[i]][k] / Au[index[k]][k];
                for (j = k; j < size + 1; ++j)
                    Au[index[i]][j] -= Au[index[k]][j] * temp;
            }       
        }
        /*Jordan elimination*/
        for (k = size - 1; k > 0; --k){
            for (i = k - 1; i >= 0; --i ){
                temp = Au[index[i]][k] / Au[index[k]][k];
                Au[index[i]][k] -= temp * Au[index[k]][k];
                Au[index[i]][size] -= temp * Au[index[k]][size];
            } 
        }
        /*solution*/
        for (k=0; k< size; ++k)
            X[k] = Au[index[k]][size] / Au[index[k]][k];
    }

    Lab3SaveOutput(X, size, 10);
    DestroyVec(X);
    DestroyMat(Au, size);
    free(index);
	return 0;	

    return(0);
}