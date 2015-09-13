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
    } else {
        odata[k] = idata[k];
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

    int *in;
    int *out;
    for (int d = 1; d < ilog2ceil(n)+1; d++) {
        in  = (d % 2 == 1) ? A : B;
        out = (d % 2 == 1) ? B : A;
        kScan<<<1, n>>>(d, out, in);
        checkCUDAError("scan");
        cudaDeviceSynchronize();
    }

    cudaMemcpy(odata, out, array_size, cudaMemcpyDeviceToHost);

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
