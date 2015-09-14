#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata, float *time) {
    thrust::device_vector<int> ivec(idata, idata+n);
    thrust::device_vector<int> ovec(odata, odata+n);

        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

    thrust::exclusive_scan(ivec.begin(), ivec.end(), ovec.begin());

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(time, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);

    thrust::copy(ovec.begin(), ovec.end(), odata);
}

}
}
