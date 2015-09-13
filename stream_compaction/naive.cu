#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void kScan(int d, int *odata, const int *idata) {
    int k = threadIdx.x;
    if (k >= (int)exp2f(d-1)) {
        odata[k] = idata[k - (int)exp2f(d-1)] + idata[k];
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
__host__ void scan(int n, int *odata, const int *idata) {
    int *A;
    int *B;
    int array_size = n * sizeof(int);

    cudaMalloc((void**) &A, array_size);
    cudaMalloc((void**) &B, array_size);
    cudaMemcpy(A, idata, array_size, cudaMemcpyHostToDevice);

    for (int i = 1; i < ilog2ceil(n)+1; i++) {
        kScan<<<1, n>>>(i, B, A);
        checkCUDAError("scan");
        cudaDeviceSynchronize();

        cudaMemcpy(A, B, array_size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(odata, A, array_size, cudaMemcpyDeviceToHost);

    // shift odata to the right for exclusive scan
    for (int i = n-1; i >= 0; i--) {
        odata[i+1] = odata[i];
    }
    odata[0] = 0;

    cudaFree(A);
    cudaFree(B);
}

}
}
