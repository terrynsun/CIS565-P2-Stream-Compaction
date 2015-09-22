#pragma once

namespace StreamCompaction {
namespace Fewer {
    void scan(int n, int *odata, const int *idata);

    int compact(int n, int *odata, const int *idata);
}
}
