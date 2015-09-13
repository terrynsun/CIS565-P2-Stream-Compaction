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
void scan(int n, int *odata, const int *idata) {
    thrust::device_vector<int> ivec(idata, idata+n);
    thrust::device_vector<int> ovec(odata, odata+n);
    thrust::exclusive_scan(ivec.begin(), ivec.end(), ovec.begin());
    thrust::copy(ovec.begin(), ovec.end(), odata);
}

}
}
