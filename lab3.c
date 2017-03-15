#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab3IO.h"
#include "timer.h"
#include <omp.h> 
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

	/*Load the datasize and verify it*/
	Lab3LoadInput(&Au, &size);
	/*Calculate the solution by serial code*/
	X = CreateVec(size);
    index = malloc(size * sizeof(int));
	
    GET_TIME(start);

    // assign the indexes in parralel
    //# pragma omp parallel for num_threads(thread_count)    
    for (i = 0; i < size; ++i){
        index[i] = i;
    }

    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
	//# pragma omp parallel private(temp, i, k, j) num_threads(thread_count)
        for (k = 0; k < size - 1; ++k){
            /*Pivoting*/

		
                temp = 0;
		j = 0;
		//printf("temp: %f\n",temp);
                # pragma omp parallel for 
                for (i = k; i < size; ++i){

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

	    {
		    if (j != k)/*swap*/{
		        i = index[j];
		        index[j] = index[k];
		        index[k] = i;
		    }
	    }
	    
            /*calculating*/
            //#pragma omp parallel for num_threads(thread_count) 

            //# pragma omp parallel for private(temp, i, k, j) num_threads(thread_count)
	    //# pragma omp parallel for
            for (i = k + 1; i < size; ++i){
		//#pragma omp critical
		//{
                	temp = Au[index[i]][k] / Au[index[k]][k];
		//}
                //printf("%f\n", temp);
                for (j = k; j < size + 1; ++j){
			//#pragma omp critical
			//{
		            Au[index[i]][j] -= Au[index[k]][j] * temp;
			    //printf("%f\n", Au[index[i]][j]);
			//}
                }
            }       
        }
        /*Jordan elimination*/
       // #pragma omp parallel num_threads(thread_count) private(k, i, temp) shared(Au, index, size) 
        for (k = size - 1; k > 0; --k){
	    //#pragma omp for
            for (i = k - 1; i >= 0; --i ){
                temp = Au[index[i]][k] / Au[index[k]][k];
                Au[index[i]][k] -= temp * Au[index[k]][k];
                Au[index[i]][size] -= temp * Au[index[k]][size];
            } 
        }
        /*solution*/
        //#pragma omp parallel for num_threads(thread_count) 
        for (k=0; k< size; ++k){
            X[k] = Au[index[k]][size] / Au[index[k]][k];
	    //printf("%d\n", (X[k]));
        }
    }
    GET_TIME(end);
    printf("%f\n",(end-start));
    printf("%d\n", size);
    Lab3SaveOutput(X, size, 10);
    DestroyVec(X);
    DestroyMat(Au, size);
    free(index);


    return(0);
}
