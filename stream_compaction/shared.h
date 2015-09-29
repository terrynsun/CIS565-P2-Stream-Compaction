#pragma once

namespace StreamCompaction {
namespace Shared {
    void dv_scan(int n, int *odata);
    void scan(int n, int *odata, int *idata);

    int compact(int n, int *odata, const int *idata);
}
}
