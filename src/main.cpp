/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <chrono>
#include <cstdio>

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"

#define BENCHMARK

void benchmarkCPU() {
    const int iterations = 100;
    int totalScan = 0;
    int totalCompactWithout = 0;
    int totalCompactWith = 0;
    printf("size, scan, compactWithoutScan, compactWithScan\n");
    for (int s = 4; s < 20; s++) {
        int SIZE = 1 << s;
        int a[SIZE];
        int b[SIZE];

        for (int i = 0; i < iterations; i++) {
            genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
            a[SIZE - 1] = 0;

            auto begin = std::chrono::high_resolution_clock::now();
            StreamCompaction::CPU::scan(SIZE, b, a);
            auto end = std::chrono::high_resolution_clock::now();
            int diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
            totalScan += diff;

            begin = std::chrono::high_resolution_clock::now();
            StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            end = std::chrono::high_resolution_clock::now();
            diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
            totalCompactWithout += diff;

            begin = std::chrono::high_resolution_clock::now();
            StreamCompaction::CPU::compactWithScan(SIZE, b, a);
            end = std::chrono::high_resolution_clock::now();
            diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
            totalCompactWith += diff;
        }
        printf("%d, %f, %f, %f\n", s,
                (float)totalScan / iterations / 1000.0,
                (float)totalCompactWithout / iterations / 1000.0,
                (float)totalCompactWith / iterations / 1000.0
                );
    }
}

void benchmarkGPUBlockSize() {
    const int iterations = 100;
    printf("block size, naive::scan, efficient::scan, efficient::compact\n");
    int SIZE = 1 << 16;
    int a[SIZE];
    int b[SIZE];
    for (int block = 2; block < 11; block++) {
        int blockSize = 1 << block;

        float totalNaive = 0;
        float totalEfficientScan = 0;
        float totalEfficientCompact = 0;
        for (int i = 0; i < iterations; i++) {
            genArray(SIZE - 1, a, 50);
            a[SIZE - 1] = 0;
            zeroArray(SIZE, b);

            float timeElapsed;

            StreamCompaction::Naive::scan(SIZE, b, a, &timeElapsed, blockSize);
            totalNaive += timeElapsed;

            StreamCompaction::Efficient::scan(SIZE, b, a, &timeElapsed, blockSize);
            totalEfficientScan += timeElapsed;

            StreamCompaction::Efficient::compact(SIZE, b, a, &timeElapsed, blockSize);
            totalEfficientCompact += timeElapsed;
        }
        printf("%d, %f, %f, %f\n", block,
                totalNaive / iterations,
                totalEfficientScan / iterations,
                totalEfficientCompact / iterations
              );
    }
}

void benchmarkGPUArraySize() {
    const int iterations = 100;
    printf("block size, naive::scan, efficient::scan, efficient::compact, thrust::scan\n");
    for (int s = 4; s < 20; s++) {
        int SIZE = 1 << s;
        int a[SIZE];
        int b[SIZE];

        int blockSize = 1 << 7;

        float totalNaive = 0;
        float totalEfficientScan = 0;
        float totalEfficientCompact = 0;
        float totalThrust = 0;
        for (int i = 0; i < iterations; i++) {
            genArray(SIZE - 1, a, 50);
            a[SIZE - 1] = 0;
            zeroArray(SIZE, b);

            float timeElapsed;

            StreamCompaction::Naive::scan(SIZE, b, a, &timeElapsed, blockSize);
            totalNaive += timeElapsed;

            StreamCompaction::Efficient::scan(SIZE, b, a, &timeElapsed, blockSize);
            totalEfficientScan += timeElapsed;

            StreamCompaction::Efficient::compact(SIZE, b, a, &timeElapsed, blockSize);
            totalEfficientCompact += timeElapsed;

            StreamCompaction::Thrust::scan(SIZE, b, a, &timeElapsed);
            totalThrust += timeElapsed;
        }
        printf("%d, %f, %f, %f, %f\n", s,
                totalNaive / iterations,
                totalEfficientScan / iterations,
                totalEfficientCompact / iterations,
                totalThrust / iterations
              );
    }
}

int main(int argc, char* argv[]) {
#ifdef BENCHMARK
    benchmarkCPU();
    benchmarkGPUBlockSize();
    benchmarkGPUArraySize();
#else
    const int SIZE = 1 << 8;
    //const int SIZE = 4;
    const int NPOT_SIZE = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT_SIZE, c, a);
    printArray(NPOT_SIZE, b, true);
    printCmpResult(NPOT_SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT_SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT_SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT_SIZE, c, a);
    printArray(NPOT_SIZE, c, true);
    printCmpResult(NPOT_SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT_SIZE, c, a);
    //printArray(NPOT_SIZE, c, true);
    printCmpResult(NPOT_SIZE, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 2);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT_SIZE;

    zeroArray(SIZE, b);
//    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
//    printArray(count, b, true);
//    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
//    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT_SIZE, c, a);
    expectedNPOT_SIZE = count;
//    printArray(count, c, true);
//    printCmpLenResult(count, expectedNPOT_SIZE, b, c);

    zeroArray(SIZE, c);
//    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
//    printArray(count, c, true);
//    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT_SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT_SIZE, b, c);

    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    genArray(SIZE - 1, a, 3);  // Leave a 0 at the end to test that edge case
    a[0] = 0;
    a[1] = 2;
    a[2] = 3;
    a[3] = 1;
    //a = { 0, 1, 2, 3 };
    //a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::sort(SIZE, c, a);
    printArray(SIZE, c, true);
    printArrayOrderResult(SIZE, c);
#endif
}
