#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void kUpSweep(int d, int *data) {
    int k = threadIdx.x;
    int exp_d  = (int)exp2f(d);
    int exp_d1 = (int)exp2f(d+1);
    if (k % exp_d1 == 0) {
        data[k + exp_d1 - 1] += data[k + exp_d - 1];
    }
}

__global__ void zeroLastElt(int n, int *odata) {
    odata[n-1] = 0;
}

__global__ void kDownSweep(int d, int *data) {
    int k = threadIdx.x;
    if (k % (int)exp2f(d+1) == 0) {
        int left  = k + (int)exp2f(d) - 1;
        int right = k + (int)exp2f(d+1) - 1;
        int t = data[left];
        data[left] = data[right];
        data[right] += t;
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int size, int *odata, const int *input) {
    int *idata;
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
        for (int j = 0; j < n; j++) {
            if (j < size) {
                idata[j] = input[j];
            } else {
                idata[j] = 0;
            }
        }
    } else {
        n = size;
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
    }

    int *A;
    int array_size = n * sizeof(int);

    cudaMalloc((void**) &A, array_size);
    cudaMemcpy(A, idata, array_size, cudaMemcpyHostToDevice);

    for (int d = 0; d < ilog2ceil(n)-1; d++) {
        kUpSweep<<<1, n>>>(d, A);
    }

    zeroLastElt<<<1, 1>>>(n, A);

    for (int d = ilog2ceil(n)-1; d >= 0; d--) {
        kDownSweep<<<1, n>>>(d, A);
    }

    cudaMemcpy(odata, A, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
