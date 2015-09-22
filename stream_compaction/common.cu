#include "common.h"

#define ERRORCHECK 1

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
#endif
}


namespace StreamCompaction {
namespace Common {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
    int k = threadIdx.x;
    if (k >= n) { return; }

    bools[k] = (idata[k] != 0) ? 1 : 0;
}

__global__ void kernScatter(int n, int *odata, int *indices, int *idata) {
    int k = threadIdx.x;
    if (k >= n) { return; }
    if (k == n-1) {
        // always take the last element
        // `compact` will adjust size appropriately
        odata[indices[k]] = idata[k];
    } else if (indices[k] != indices[k+1]) {
        odata[indices[k]] = idata[k];
    }
}

}
}
