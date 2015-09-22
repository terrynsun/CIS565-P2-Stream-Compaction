#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Fewer {

int BLOCK_SIZE = (2 << 7);

__global__ void kUpSweep(int d, int *data, int stride, int n) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    int exp_d  = (int)exp2f(d);
    int exp_d1 = (int)exp2f(d+1);

    k = k * exp_d1;

    if (k > n) { return; }
    if (k + exp_d1 - 1 > n) { return; }

    data[k + exp_d1 - 1] += data[k + exp_d - 1];
}

__global__ void kDownSweep(int d, int *data, int n) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    //if (k % (int)exp2f(d+1) == 0) {
    int exp_d1 = (int)exp2f(d+1);
    k = k * exp_d1;
    if (k < n) {
        int left  = k + (int)exp2f(d) - 1;
        int right = k + (int)exp2f(d+1) - 1;
        int t = data[left];

        data[left] = data[right];
        data[right] += t;
    }
}

/*
 * In-place scan on `dev_data`, which must be a device memory pointer.
 */
void dv_scan(int n, int *dev_data) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int ilog2 = ilog2ceil(n);

    for (int d = 0; d < ilog2-1; d++) {
        int stride = exp2f(d+1);
        int total = ((float) n / (float) stride);
        numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kUpSweep<<<numBlocks, BLOCK_SIZE>>>(d, dev_data, stride, n);
        cudaDeviceSynchronize();
        checkCUDAError("scan (1)");
    }

    int z = 0;
    cudaMemcpy(&dev_data[n-1], &z, sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudamemcpy");

    numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int d = ilog2-1; d >= 0; d--) {
        int stride = exp2f(d+1);
        int total = ((float) n / (float) stride);
        numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kDownSweep<<<numBlocks, BLOCK_SIZE>>>(d, dev_data, n);
        checkCUDAError("scan (2)");
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

    dv_scan(n, dv_idata);

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
int compact(int size, int *odata, const int *input) {
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

    StreamCompaction::Common::kernMapToBoolean<<<numBlocks, BLOCK_SIZE>>>(n, dev_indices, dev_idata);

    int last;
    cudaMemcpy(&last, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    dv_scan(n, dev_indices);
    int streamSize;
    cudaMemcpy(&streamSize, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    StreamCompaction::Common::kernScatter<<<numBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_indices, dev_idata);

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
