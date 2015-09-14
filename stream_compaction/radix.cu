#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Radix {

/*
 * Get INVERTED `d`th digit of odata[k].
 */
__global__ void kDigit(int n, int d, int *dv_odata, const int *dv_idata) {
    int k = threadIdx.x;
    if (k >= n) { return; }
    dv_odata[k] = (dv_idata[k] & (1 << d)) > 0 ? 1 : 0;
}

__global__ void kInvert(int n, int *odata, const int *idata) {
    int k = threadIdx.x;
    if (k >= n) { return; }
    odata[k] = idata[k] == 0 ? 1 : 0;
}

__global__ void kMapToIndex(int n, int *odata, int *b, int *f_indices, int pivot) {
    int k = threadIdx.x;
    if (k >= n) { return; }
    odata[k] = (b[k] == 1) ? (k - f_indices[k] + pivot) : f_indices[k];
}

/*
 * Implement split on device memory.
 * Returns totalFalses (eg. the split point).
 */
__host__ int split(int n, int d, int *dv_odata, int *dv_idata) {
    printf("---- split %d %d ----\n", n, d);
    int array_size = n * sizeof(int);
    int *TMP = (int*)malloc(array_size);

    int *b;
    int *e;
    int *t;
    int *indices;
    cudaMalloc((void**) &b, array_size);
    cudaMalloc((void**) &e, array_size);
    cudaMalloc((void**) &t, array_size);
    cudaMalloc((void**) &indices, array_size);

    kDigit<<<1, n>>>(n, d, b, dv_idata); // b
        printf("b: ");
        cudaMemcpy(TMP, b, array_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) { printf("%d\t", TMP[i]); } printf("\n");
    kInvert<<<1, n>>>(n, e, b); // e
        printf("e: ");
        cudaMemcpy(TMP, e, array_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) { printf("%d\t", TMP[i]); } printf("\n");

    int lastElt;
    cudaMemcpy(&lastElt, e + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    StreamCompaction::Efficient::dv_scan(n, e); // f IN PLACE OF e

    int totalFalses;
    cudaMemcpy(&totalFalses, e + n-1, sizeof(int), cudaMemcpyDeviceToHost);
    totalFalses += lastElt;

    printf("totalFalses = %d\n", totalFalses);

    kMapToIndex<<<1, n>>>(n, indices, b, e, totalFalses);
        printf("indices: ");
        cudaMemcpy(TMP, indices, array_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) { printf("%d\t", TMP[i]); } printf("\n");

    StreamCompaction::Common::kernScatter<<<1, n>>>(n, dv_odata, indices, dv_idata); // scatter
        printf("scattered: ");
        cudaMemcpy(TMP, dv_odata, array_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) { printf("%d\t", TMP[i]); } printf("\n");

    cudaFree(b);
    return totalFalses;
}

int testArrayOrder(int n, int *a) {
    for (int i = 0; i < n-1; i++) {
        if (a[i] > a[i+1]) {
            return 1;
        }
    }
    return 0;
}

/*
 * odata and idata are device memory points.
 */
__host__ void sortRecursive(int n, int d, int dmax, int *odata, int *idata) {
    if (d >= dmax) { return; }
    int pivot = split(n, d, odata, idata);
    //sortRecursive(n, d+1, dmax, odata, odata);
    //if (pivot != 0) {
    //    sortRecursive(pivot, d+1, dmax, odata, odata);
    //}
    //if (pivot != n) {
    //    sortRecursive(n-pivot, d+1, dmax, odata+n, odata+n);
    //}
}

__host__ void sortRecursive2(int n, int d, int dmax, int *odata, int *idata) {
    if (d <= 0) { return; }
    int pivot = split(n, d, odata, idata);
    if (pivot != 0) {
        sortRecursive(pivot, d-1, dmax, odata, odata);
    }
    if (pivot != n) {
        sortRecursive(n-pivot, d-1, dmax, odata+n, odata+n);
    }
}

__host__ void sort(int n, int *odata, const int *idata) {
    int max = idata[0];
    for (int i = 0; i < n; i++) {
        if (idata[i] > max) {
            max = idata[i];
        }
    }
    int maxDigits = ilog2ceil(max);

    int *dv_odata;
    int *dv_idata;
    int array_size = n * sizeof(int);

    cudaMalloc((void**) &dv_odata, array_size);
    cudaMalloc((void**) &dv_idata, array_size);
    cudaMemcpy(dv_idata, idata, array_size, cudaMemcpyHostToDevice);

    //sortRecursive(n, 0, maxDigits, dv_odata, dv_idata);
    sortRecursive2(n, 0, maxDigits, dv_odata, dv_idata);

    cudaMemcpy(odata, dv_odata, array_size, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < n; i++) { printf("%d\t%d\n", idata[i], odata[i]); }

    cudaFree(dv_odata);
    cudaFree(dv_idata);
}

}
}
