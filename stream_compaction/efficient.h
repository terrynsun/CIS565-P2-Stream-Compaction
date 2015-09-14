#pragma once

namespace StreamCompaction {
namespace Efficient {
    void dv_scan(int n, int *dev_idata);
    void scan(int size, int *odata, const int *input, float *time, int blockSize);

    int compact(int n, int *odata, const int *idata, float *time, int blockSize);
}
}
