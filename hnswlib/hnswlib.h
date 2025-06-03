#pragma once

// // https://github.com/nmslib/hnswlib/pull/508
// // This allows others to provide their own error stream (e.g. RcppHNSW)
// #ifndef HNSWLIB_ERR_OVERRIDE
//   #define HNSWERR std::cerr
// #else
//   #define HNSWERR HNSWLIB_ERR_OVERRIDE
// #endif

// #ifndef NO_MANUAL_VECTORIZATION
// #if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
// #define USE_SSE
// #ifdef __AVX__
// #define USE_AVX
// #ifdef __AVX512F__
// #define USE_AVX512
// #endif
// #endif
// #endif
// #endif

// #if defined(USE_AVX) || defined(USE_SSE)
// #ifdef _MSC_VER
// #include <intrin.h>
// #include <stdexcept>
// static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
//     __cpuidex(out, eax, ecx);
// }
// static __int64 xgetbv(unsigned int x) {
//     return _xgetbv(x);
// }
// #else
// #include <x86intrin.h>
// #include <cpuid.h>
// #include <stdint.h>
// static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
//     __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
// }
// static uint64_t xgetbv(unsigned int index) {
//     uint32_t eax, edx;
//     __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
//     return ((uint64_t)edx << 32) | eax;
// }
// #endif

// #if defined(USE_AVX512)
// #include <immintrin.h>
// #endif

// #if defined(__GNUC__)
// #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
// #define PORTABLE_ALIGN64 __attribute__((aligned(64)))
// #else
// #define PORTABLE_ALIGN32 __declspec(align(32))
// #define PORTABLE_ALIGN64 __declspec(align(64))
// #endif

// // Adapted from https://github.com/Mysticial/FeatureDetector
// #define _XCR_XFEATURE_ENABLED_MASK  0

// static bool AVXCapable() {
//     int cpuInfo[4];

//     // CPU support
//     cpuid(cpuInfo, 0, 0);
//     int nIds = cpuInfo[0];

//     bool HW_AVX = false;
//     if (nIds >= 0x00000001) {
//         cpuid(cpuInfo, 0x00000001, 0);
//         HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
//     }

//     // OS support
//     cpuid(cpuInfo, 1, 0);

//     bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
//     bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

//     bool avxSupported = false;
//     if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
//         uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
//         avxSupported = (xcrFeatureMask & 0x6) == 0x6;
//     }
//     return HW_AVX && avxSupported;
// }

// static bool AVX512Capable() {
//     if (!AVXCapable()) return false;

//     int cpuInfo[4];

//     // CPU support
//     cpuid(cpuInfo, 0, 0);
//     int nIds = cpuInfo[0];

//     bool HW_AVX512F = false;
//     if (nIds >= 0x00000007) {  //  AVX512 Foundation
//         cpuid(cpuInfo, 0x00000007, 0);
//         HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
//     }

//     // OS support
//     cpuid(cpuInfo, 1, 0);

//     bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
//     bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

//     bool avx512Supported = false;
//     if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
//         uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
//         avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
//     }
//     return HW_AVX512F && avx512Supported;
// }
// #endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswmips {
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnswmips::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

template<typename dist_t>
class BaseSearchStopCondition {
 public:
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

template<typename MTYPE>
class SpaceInterface {
 public:
    // virtual void search(void *);
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void *get_dist_func_param() = 0;

    float norm(const float *a, unsigned size) const {
        float result = 0;
    #ifdef __GNUC__
    #ifdef __AVX__
    #define AVX_L2NORM(addr, dest, tmp) \
      tmp = _mm256_loadu_ps(addr);      \
      tmp = _mm256_mul_ps(tmp, tmp);    \
      dest = _mm256_add_ps(dest, tmp);
    
        __m256 sum;
        __m256 l0, l1;
        unsigned D = (size + 7) & ~7U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = a;
        const float *e_l = l + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    
        sum = _mm256_loadu_ps(unpack);
        if (DR) {
          AVX_L2NORM(e_l, sum, l0);
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
          AVX_L2NORM(l, sum, l0);
          AVX_L2NORM(l + 8, sum, l1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
                 unpack[5] + unpack[6] + unpack[7];
    #else
    #ifdef __SSE2__
    #define SSE_L2NORM(addr, dest, tmp) \
      tmp = _mm_loadu_ps(addr);      \
      tmp = _mm_mul_ps(tmp, tmp);    \
      dest = _mm_add_ps(dest, tmp);
    
        __m128 sum;
        __m128 l0, l1, l2, l3;
        unsigned D = (size + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = a;
        const float *e_l = l + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};
    
        sum = _mm_load_ps(unpack);
        switch (DR) {
          case 12:
            SSE_L2NORM(e_l + 8, sum, l2);
          case 8:
            SSE_L2NORM(e_l + 4, sum, l1);
          case 4:
            SSE_L2NORM(e_l, sum, l0);
          default:
            break;
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
          SSE_L2NORM(l, sum, l0);
          SSE_L2NORM(l + 4, sum, l1);
          SSE_L2NORM(l + 8, sum, l2);
          SSE_L2NORM(l + 12, sum, l3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
    #else
        float dot0, dot1, dot2, dot3;
        const float* last = a + size;
        const float* unroll_group = last - 3;
    
        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
          dot0 = a[0] * a[0];
          dot1 = a[1] * a[1];
          dot2 = a[2] * a[2];
          dot3 = a[3] * a[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
          result += (*a) * (*a);
          a++;
        }
    #endif
    #endif
    #endif
        return result;
    }

    virtual ~SpaceInterface() {}
};

template<typename dist_t>
class AlgorithmInterface {
 public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswmips

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
