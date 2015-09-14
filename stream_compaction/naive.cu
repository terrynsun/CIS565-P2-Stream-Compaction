#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

int BLOCK_SIZE = 128;

__global__ void kScan(int d, int *odata, const int *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k >= (int)exp2f(d-1)) {
        odata[k] = idata[k - (int)exp2f(d-1)] + idata[k];
    } else {
        odata[k] = idata[k];
    }
}

__global__ void kShift(int n, int *odata, int *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k >= n) { return; }
    if (k == 0) {
        odata[0] = 0;
    } else {
        odata[k] = idata[k-1];
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
__host__ void scan(int n, int *odata, const int *idata) {
    int array_size = n * sizeof(int);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *A;
    int *B;

    cudaMalloc((void**) &A, array_size);
    cudaMalloc((void**) &B, array_size);
    cudaMemcpy(A, idata, array_size, cudaMemcpyHostToDevice);

    int *in;
    int *out;

    for (int d = 1; d < ilog2ceil(n)+1; d++) {
        in  = (d % 2 == 1) ? A : B;
        out = (d % 2 == 1) ? B : A;
        kScan<<<numBlocks, BLOCK_SIZE>>>(d, out, in);
        checkCUDAError("scan");
    }

    // shift odata to the right for exclusive scan
    kShift<<<numBlocks, BLOCK_SIZE>>>(n, in, out);

    cudaMemcpy(odata, in, array_size, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
}

}
}
