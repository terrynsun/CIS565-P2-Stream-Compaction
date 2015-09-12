#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        odata[i] = total;
        total += idata[i];
    }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (idata[i] != 0) {
            odata[count++] = idata[i];
        }
    }
    return count;
}

/**
  * CPU scatter algorithm.
  *
  * @returns the number of elements remaining.
  */
int scatter(int n, int *odata, const int *indices, const int *input) {
    for (int i = 0; i < n; i++) {
        odata[indices[i]] = input[i];
    }
    return indices[n-1];
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    int *predicate_data = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        predicate_data[i] = idata[i] == 0 ? 0 : 1;
    }

    int *scan_data = (int *)malloc(n * sizeof(int));
    scan(n, scan_data, predicate_data);

    int count = scatter(n, odata, scan_data, idata);

    free(predicate_data);
    free(scan_data);

    return count;
}

}
}
