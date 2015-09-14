#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

int BLOCK_SIZE = (2 << 7);

__global__ void kUpSweep(int d, int *data) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    int exp_d  = (int)exp2f(d);
    int exp_d1 = (int)exp2f(d+1);
    if (k % exp_d1 == 0) {
        data[k + exp_d1 - 1] += data[k + exp_d - 1];
    }
}

__global__ void kDownSweep(int d, int *data) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k % (int)exp2f(d+1) == 0) {
        int left  = k + (int)exp2f(d) - 1;
        int right = k + (int)exp2f(d+1) - 1;
        int t = data[left];
        data[left] = data[right];
        data[right] += t;
    }
}

/*
 * In-place scan on `dev_idata`, which must be a device memory pointer.
 */
void dv_scan(int n, int *dev_idata) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int d = 0; d < ilog2ceil(n)-1; d++) {
        kUpSweep<<<numBlocks, BLOCK_SIZE>>>(d, dev_idata);
        checkCUDAError("scan");
    }

    int z = 0;
    cudaMemcpy(&dev_idata[n-1], &z, sizeof(int), cudaMemcpyHostToDevice);

    for (int d = ilog2ceil(n)-1; d >= 0; d--) {
        kDownSweep<<<numBlocks, BLOCK_SIZE>>>(d, dev_idata);
        checkCUDAError("scan");
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int size, int *odata, const int *input, float *time, int blockSize) {
    BLOCK_SIZE = blockSize;
    int *idata;
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
        for (int j = size; j < n; j++) {
            idata[j] = 0;
        }
    } else {
        n = size;
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
    }


    int array_size = n * sizeof(int);
    int *dv_idata;

    cudaMalloc((void**) &dv_idata, array_size);
    cudaMemcpy(dv_idata, idata, array_size, cudaMemcpyHostToDevice);

        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

    dv_scan(n, dv_idata);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(time, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);

    cudaMemcpy(odata, dv_idata, array_size, cudaMemcpyDeviceToHost);
    cudaFree(dv_idata);
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
int compact(int size, int *odata, const int *input, float *time, int blockSize) {
    BLOCK_SIZE = blockSize;

    int *idata;
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
        for (int j = size; j < n; j++) {
            idata[j] = 0;
        }
    } else {
        n = size;
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
    }

    int *dev_indices;
    int *dev_odata;
    int *dev_idata;
    int array_size = n * sizeof(int);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void**) &dev_indices, array_size);
    cudaMalloc((void**) &dev_odata, array_size);

    cudaMalloc((void**) &dev_idata, array_size);
    cudaMemcpy(dev_idata, idata, array_size, cudaMemcpyHostToDevice);

        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

    StreamCompaction::Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_indices, dev_idata);

    int last;
    cudaMemcpy(&last, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    dv_scan(n, dev_indices);
    int streamSize;
    cudaMemcpy(&streamSize, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    StreamCompaction::Common::kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_indices, dev_idata);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(time, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);

    cudaMemcpy(odata, dev_odata, array_size, cudaMemcpyDeviceToHost);

    // The kernel always copies the last elt.
    // Adjust the size to include it if desired.
    if (last == 1) {
        streamSize++;
    }

    return streamSize;
}

}
}
