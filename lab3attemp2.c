#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab3IO.h"
#include "timer.h"
//****************************************************
// modified from the serial solution in serialTester 
//****************************************************
int main(int argc, char* argv[]){
     double start,end;
	int i, j, k, size;
	double** Au;
	double* X;
	double temp, error, Xnorm;
	int* index;
	FILE* fp;

    	//get the number of threads
    	int thread_count = atoi(argv[1]);
	omp_set_num_threads(thread_count);

	/*Load the datasize and verify it*/
	Lab3LoadInput(&Au, &size);
	/*Calculate the solution by serial code*/
	X = CreateVec(size);
    index = malloc(size * sizeof(int));

    // assign the indexes in parralel
    GET_TIME(start); 
    # pragma omp for
    for (i = 0; i < size; ++i){
        index[i] = i;
    }

    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
		// #pragma omp parallel
    	for(k = 0; k < size - 1; ++k) {
		
		    /*Pivoting*/
		    temp = 0;
		    j = 0;
		    # pragma omp parallel for private(i)
		    for (i = k; i < size; ++i) {
		       if (temp < Au[index[i]][k] * Au[index[i]][k]){
                     # pragma omp critical
	                 { 
	                    if (temp < Au[index[i]][k] * Au[index[i]][k]){
	                        temp = Au[index[i]][k] * Au[index[i]][k];
	                        j = i;
	                    }
	                 }
				}
            }

		 //    #pragma omp single
			// {
		    if (j != k)/*swap*/{
				i = index[j];
				index[j] = index[k];
				index[k] = i;
		    }
			// }
		    
		    /*calculating*/
		    #pragma omp parallel for private(i,temp,j)
		    for (i = k + 1; i < size; ++i){
		        temp = Au[index[i]][k] / Au[index[k]][k];
				//#pragma omp for
		        for (j = k; j < size + 1; ++j) {
		            Au[index[i]][j] -= Au[index[k]][j] * temp;
				}
		    }   
		    // #pragma omp single
		    // {  
		    // 	k++;
		    // }
    	}
        /*Jordan elimination*/
        #pragma omp parallel for private(temp, k, i)
	    for (k = size - 1; k > 0; --k){
	    // #pragma omp parallel for private(temp)
	        for (i = k - 1; i >= 0; --i ){
	            temp = Au[index[i]][k] / Au[index[k]][k];
	            Au[index[i]][k] -= temp * Au[index[k]][k];
	            Au[index[i]][size] -= temp * Au[index[k]][size];
	        } 
	    }
	        /*solution*/
		#pragma omp parallel for
	        for (k=0; k< size; ++k) {
	            X[k] = Au[index[k]][size] / Au[index[k]][k];
		    printf("%e\n", X[k]);
		}
    }
    GET_TIME(end);
    printf("%f\n",(end-start));
    printf("%d\n", size);
    Lab3SaveOutput(X, size, (end-start));
    DestroyVec(X);
    DestroyMat(Au, size);
    free(index);
	return 0;	

    return(0);
}
