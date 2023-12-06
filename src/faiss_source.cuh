#include <limits>

namespace faiss {
using idx_t = int64_t;
namespace gpu {

// We validate this against the actual architecture in device initialization
constexpr int kWarpSize = 32;

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {
#if CUDA_VERSION >= 9000
    __syncwarp();
#else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
#endif
}
__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

template <typename T>
constexpr __host__ __device__ int log2(T n, int p = 0) {
    return (n <= 1) ? p : log2(n / 2, p + 1);
}

static_assert(log2(2) == 1, "log2");
static_assert(log2(3) == 1, "log2");
static_assert(log2(4) == 2, "log2");

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
    return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
    return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));
}



template <typename T>
inline __device__ T
shfl_down(const T val, unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(0xffffffff, val, delta, width);
#else
    return __shfl_down(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
    return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(
        T* const val,
        int laneMask,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_xor(v, laneMask, width);
}



#define FAISS_ASSERT_FMT(X, FMT, ...)                    \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__,                            \
                    __VA_ARGS__);                        \
            abort();                                     \
        }                                                \
    } while (false)

#define CUDA_VERIFY(X)                      \
    do {                                    \
        auto err__ = (X);                   \
        FAISS_ASSERT_FMT(                   \
                err__ == cudaSuccess,       \
                "CUDA error %d %s",         \
                (int)err__,                 \
                cudaGetErrorString(err__)); \
    } while (0)

#define CUDA_TEST_ERROR()                \
    do {                                 \
        CUDA_VERIFY(cudaGetLastError()); \
    } while (0)


template <typename T>
struct Limits {};

constexpr int kIntMax = std::numeric_limits<int>::max();
constexpr int kIntMin = std::numeric_limits<int>::lowest();

template <>
struct Limits<int> {
    static __device__ __host__ inline int getMin() {
        return kIntMin;
    }
    static __device__ __host__ inline int getMax() {
        return kIntMax;
    }
};

constexpr idx_t kIdxTMax = std::numeric_limits<idx_t>::max();
constexpr idx_t kIdxTMin = std::numeric_limits<idx_t>::lowest();

template <>
struct Limits<idx_t> {
    static __device__ __host__ inline idx_t getMin() {
        return kIdxTMin;
    }
    static __device__ __host__ inline idx_t getMax() {
        return kIdxTMax;
    }
};

constexpr uint64_t kUint64TMax = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kUint64TMin = std::numeric_limits<uint64_t>::lowest();

template <>
struct Limits<uint64_t> {
    static __device__ __host__ inline uint64_t getMin() {
        return kUint64TMin;
    }
    static __device__ __host__ inline uint64_t getMax() {
        return kUint64TMax;
    }
};

template <typename T>
inline __device__ void swap(bool swap, T& x, T& y) {
    T tmp = x;
    x = swap ? y : x;
    y = swap ? tmp : y;
}

template <typename T>
inline __device__ void assign(bool assign, T& x, T y) {
    x = assign ? y : x;
}
// This function merges kWarpSize / 2L lists in parallel using warp
// shuffles.
// It works on at most size-16 lists, as we need 32 threads for this
// shuffle merge.
//
// If IsBitonic is false, the first stage is reversed, so we don't
// need to sort directionally. It's still technically a bitonic sort.
template <
        typename K,
        typename V,
        int L,
        bool Dir,
        typename Comp,
        bool IsBitonic>
inline __device__ void warpBitonicMergeLE16(K& k, V& v) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(L <= kWarpSize / 2, "merge list size must be <= 16");

    int laneId = getLaneId();

    if (!IsBitonic) {
        // Reverse the first comparison stage.
        // For example, merging a list of size 8 has the exchanges:
        // 0 <-> 15, 1 <-> 14, ...
        K otherK = shfl_xor(k, 2 * L - 1);
        // V otherV = shfl_xor(v, 2 * L - 1);

        // Whether we are the lesser thread in the exchange
        bool small = !(laneId & L);

        if (Dir) {
            // See the comment above how performing both of these
            // comparisons in the warp seems to win out over the
            // alternatives in practice
            bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
            assign(s, k, otherK);
            // assign(s, v, otherV);

        } else {
            bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
            assign(s, k, otherK);
            // assign(s, v, otherV);
        }
    }

#pragma unroll
    for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
        K otherK = shfl_xor(k, stride);
        // V otherV = shfl_xor(v, stride);

        // Whether we are the lesser thread in the exchange
        bool small = !(laneId & stride);

        if (Dir) {
            bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
            assign(s, k, otherK);
            // assign(s, v, otherV);

        } else {
            bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
            assign(s, k, otherK);
            // assign(s, v, otherV);
        }
    }
}

// Template for performing a bitonic merge of an arbitrary set of
// registers
template <
        typename K,
        typename V,
        int N,
        bool Dir,
        typename Comp,
        bool Low,
        bool Pow2>
struct BitonicMergeStep {};

//
// Power-of-2 merge specialization
//

// All merges eventually call this
template <typename K, typename V, bool Dir, typename Comp, bool Low>
struct BitonicMergeStep<K, V, 1, Dir, Comp, Low, true> {
    static inline __device__ void merge(K k[1], V v[1]) {
        // Use warp shuffles
        warpBitonicMergeLE16<K, V, 16, Dir, Comp, true>(k[0], v[0]);
    }
};

template <typename K, typename V, int N, bool Dir, typename Comp, bool Low>
struct BitonicMergeStep<K, V, N, Dir, Comp, Low, true> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(isPowerOf2(N), "must be power of 2");
        static_assert(N > 1, "must be N > 1");

#pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            K& ka = k[i];
            // V& va = v[i];

            K& kb = k[i + N / 2];
            // V& vb = v[i + N / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
            // swap(s, va, vb);
        }

        {
            K newK[N / 2];
            V newV[N / 2];

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                newK[i] = k[i];
                // newV[i] = v[i];
            }

            BitonicMergeStep<K, V, N / 2, Dir, Comp, true, true>::merge(
                    newK, newV);

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                k[i] = newK[i];
                // v[i] = newV[i];
            }
        }

        {
            K newK[N / 2];
            V newV[N / 2];

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                newK[i] = k[i + N / 2];
                // newV[i] = v[i + N / 2];
            }

            BitonicMergeStep<K, V, N / 2, Dir, Comp, false, true>::merge(
                    newK, newV);

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                k[i + N / 2] = newK[i];
                // v[i + N / 2] = newV[i];
            }
        }
    }
};

//
// Non-power-of-2 merge specialization
//

// Low recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStep<K, V, N, Dir, Comp, true, false> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(!isPowerOf2(N), "must be non-power-of-2");
        static_assert(N >= 3, "must be N >= 3");

        constexpr int kNextHighestPowerOf2 = nextHighestPowerOf2(N);

#pragma unroll
        for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
            K& ka = k[i];
            // V& va = v[i];

            K& kb = k[i + kNextHighestPowerOf2 / 2];
            // V& vb = v[i + kNextHighestPowerOf2 / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
            // swap(s, va, vb);
        }

        constexpr int kLowSize = N - kNextHighestPowerOf2 / 2;
        constexpr int kHighSize = kNextHighestPowerOf2 / 2;
        {
            K newK[kLowSize];
            V newV[kLowSize];

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                newK[i] = k[i];
                // newV[i] = v[i];
            }

            constexpr bool kLowIsPowerOf2 =
                    isPowerOf2(N - kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kLowIsPowerOf2 = isPowerOf2(kLowSize);
            BitonicMergeStep<
                    K,
                    V,
                    kLowSize,
                    Dir,
                    Comp,
                    true, // low
                    kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                k[i] = newK[i];
                // v[i] = newV[i];
            }
        }

        {
            K newK[kHighSize];
            V newV[kHighSize];

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                newK[i] = k[i + kLowSize];
                // newV[i] = v[i + kLowSize];
            }

            constexpr bool kHighIsPowerOf2 =
                    isPowerOf2(kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kHighIsPowerOf2 =
            //      isPowerOf2(kHighSize);
            BitonicMergeStep<
                    K,
                    V,
                    kHighSize,
                    Dir,
                    Comp,
                    false, // high
                    kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                k[i + kLowSize] = newK[i];
                // v[i + kLowSize] = newV[i];
            }
        }
    }
};

// High recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStep<K, V, N, Dir, Comp, false, false> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(!isPowerOf2(N), "must be non-power-of-2");
        static_assert(N >= 3, "must be N >= 3");

        constexpr int kNextHighestPowerOf2 = nextHighestPowerOf2(N);

#pragma unroll
        for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
            K& ka = k[i];

            K& kb = k[i + kNextHighestPowerOf2 / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
        }

        constexpr int kLowSize = kNextHighestPowerOf2 / 2;
        constexpr int kHighSize = N - kNextHighestPowerOf2 / 2;
        {
            K newK[kLowSize];
            V newV[kLowSize];

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                newK[i] = k[i];
                // newV[i] = v[i];
            }

            constexpr bool kLowIsPowerOf2 =
                    isPowerOf2(kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kLowIsPowerOf2 = isPowerOf2(kLowSize);
            BitonicMergeStep<
                    K,
                    V,
                    kLowSize,
                    Dir,
                    Comp,
                    true, // low
                    kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                k[i] = newK[i];
            }
        }

        {
            K newK[kHighSize];
            V newV[kHighSize];

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                newK[i] = k[i + kLowSize];
            }

            constexpr bool kHighIsPowerOf2 =
                    isPowerOf2(N - kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kHighIsPowerOf2 =
            //      isPowerOf2(kHighSize);
            BitonicMergeStep<
                    K,
                    V,
                    kHighSize,
                    Dir,
                    Comp,
                    false, // high
                    kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                k[i + kLowSize] = newK[i];
            }
        }
    }
};

/// Merges two sets of registers across the warp of any size;
/// i.e., merges a sorted k/v list of size kWarpSize * N1 with a
/// sorted k/v list of size kWarpSize * N2, where N1 and N2 are any
/// value >= 1
template <
        typename K,
        typename V,
        int N1,
        int N2,
        bool Dir,
        typename Comp,
        bool FullMerge = true>
inline __device__ void warpMergeAnyRegisters(
        K k1[N1],
        V v1[N1],
        K k2[N2],
        V v2[N2]) {
    constexpr int kSmallestN = N1 < N2 ? N1 : N2;

#pragma unroll
    for (int i = 0; i < kSmallestN; ++i) {
        K& ka = k1[N1 - 1 - i];

        K& kb = k2[i];

        K otherKa;

        if (FullMerge) {
            // We need the other values
            otherKa = shfl_xor(ka, kWarpSize - 1);
        }

        K otherKb = shfl_xor(kb, kWarpSize - 1);

        // ka is always first in the list, so we needn't use our lane
        // in this comparison
        bool swapa = Dir ? Comp::gt(ka, otherKb) : Comp::lt(ka, otherKb);
        assign(swapa, ka, otherKb);

        // kb is always second in the list, so we needn't use our lane
        // in this comparison
        if (FullMerge) {
            bool swapb = Dir ? Comp::lt(kb, otherKa) : Comp::gt(kb, otherKa);
            assign(swapb, kb, otherKa);

        } else {
            // We don't care about updating elements in the second list
        }
    }

    BitonicMergeStep<K, V, N1, Dir, Comp, true, isPowerOf2(N1)>::merge(
            k1, v1);
    if (FullMerge) {
        // Only if we care about N2 do we need to bother merging it fully
        BitonicMergeStep<K, V, N2, Dir, Comp, false, isPowerOf2(N2)>::
                merge(k2, v2);
    }
}

// Recursive template that uses the above bitonic merge to perform a
// bitonic sort
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicSortStep {
    static inline __device__ void sort(K k[N], V v[N]) {
        static_assert(N > 1, "did not hit specialized case");

        // Sort recursively
        constexpr int kSizeA = N / 2;
        constexpr int kSizeB = N - kSizeA;

        K aK[kSizeA];
        V aV[kSizeA];

#pragma unroll
        for (int i = 0; i < kSizeA; ++i) {
            aK[i] = k[i];
        }

        BitonicSortStep<K, V, kSizeA, Dir, Comp>::sort(aK, aV);

        K bK[kSizeB];
        V bV[kSizeB];

#pragma unroll
        for (int i = 0; i < kSizeB; ++i) {
            bK[i] = k[i + kSizeA];
        }

        BitonicSortStep<K, V, kSizeB, Dir, Comp>::sort(bK, bV);

        // Merge halves
        warpMergeAnyRegisters<K, V, kSizeA, kSizeB, Dir, Comp>(aK, aV, bK, bV);

#pragma unroll
        for (int i = 0; i < kSizeA; ++i) {
            k[i] = aK[i];
        }

#pragma unroll
        for (int i = 0; i < kSizeB; ++i) {
            k[i + kSizeA] = bK[i];
        }
    }
};

// Single warp (N == 1) sorting specialization
template <typename K, typename V, bool Dir, typename Comp>
struct BitonicSortStep<K, V, 1, Dir, Comp> {
    static inline __device__ void sort(K k[1], V v[1]) {
        // Update this code if this changes
        // should go from 1 -> kWarpSize in multiples of 2
        static_assert(kWarpSize == 32, "unexpected warp size");

        warpBitonicMergeLE16<K, V, 1, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 2, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 4, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 8, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 16, Dir, Comp, false>(k[0], v[0]);
    }
};

/// Sort a list of kWarpSize * N elements in registers, where N is an
/// arbitrary >= 1
template <typename K, typename V, int N, bool Dir, typename Comp>
inline __device__ void warpSortAnyRegisters(K k[N], V v[N]) {
    BitonicSortStep<K, V, N, Dir, Comp>::sort(k, v);
}



// Merge pairs of lists smaller than blockDim.x (NumThreads)
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool AllThreads,
        bool Dir,
        typename Comp,
        bool FullMerge>
inline __device__ void blockMergeSmall(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(
            isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L <= NumThreads, "merge list size must be <= NumThreads");

    // Which pair of lists we are merging
    int mergeId = threadIdx.x / L;

    // Which thread we are within the merge
    int tid = threadIdx.x % L;

    // listK points to a region of size N * 2 * L
    listK += 2 * L * mergeId;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
    int pos = L - 1 - tid;
    int stride = 2 * tid + 1;

    if (AllThreads || (threadIdx.x < N * L)) {
        K ka = listK[pos];
        K kb = listK[pos + stride];

        bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
        int pos = 2 * tid - (tid & (stride - 1));

        if (AllThreads || (threadIdx.x < N * L)) {
            K ka = listK[pos];
            K kb = listK[pos + stride];

            bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

// Merge pairs of sorted lists larger than blockDim.x (NumThreads)
template <
        int NumThreads,
        typename K,
        typename V,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
inline __device__ void blockMergeLarge(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(L >= kWarpSize, "merge list size must be >= 32");
    static_assert(
            isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L >= NumThreads, "merge list size must be >= NumThreads");

    // For L > NumThreads, each thread has to perform more work
    // per each stride.
    constexpr int kLoopPerThread = L / NumThreads;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
#pragma unroll
    for (int loop = 0; loop < kLoopPerThread; ++loop) {
        int tid = loop * NumThreads + threadIdx.x;
        int pos = L - 1 - tid;
        int stride = 2 * tid + 1;

        K ka = listK[pos];
        K kb = listK[pos + stride];

        bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

    constexpr int kSecondLoopPerThread =
            FullMerge ? kLoopPerThread : kLoopPerThread / 2;

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int loop = 0; loop < kSecondLoopPerThread; ++loop) {
            int tid = loop * NumThreads + threadIdx.x;
            int pos = 2 * tid - (tid & (stride - 1));

            K ka = listK[pos];
            K kb = listK[pos + stride];

            bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

/// Class template to prevent static_assert from firing for
/// mixing smaller/larger than block cases
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool SmallerThanBlock,
        bool FullMerge>
struct BlockMerge {};

/// Merging lists smaller than a block
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, true, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        constexpr int kNumParallelMerges = NumThreads / L;
        constexpr int kNumIterations = N / kNumParallelMerges;

        static_assert(L <= NumThreads, "list must be <= NumThreads");
        static_assert(
                (N < kNumParallelMerges) ||
                        (kNumIterations * kNumParallelMerges == N),
                "improper selection of N and L");

        if (N < kNumParallelMerges) {
            // We only need L threads per each list to perform the merge
            blockMergeSmall<
                    NumThreads,
                    K,
                    V,
                    N,
                    L,
                    false,
                    Dir,
                    Comp,
                    FullMerge>(listK, listV);
        } else {
            // All threads participate
#pragma unroll
            for (int i = 0; i < kNumIterations; ++i) {
                int start = i * kNumParallelMerges * 2 * L;

                blockMergeSmall<
                        NumThreads,
                        K,
                        V,
                        N,
                        L,
                        true,
                        Dir,
                        Comp,
                        FullMerge>(listK + start, listV + start);
            }
        }
    }
};

/// Merging lists larger than a block
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, false, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        // Each pair of lists is merged sequentially
#pragma unroll
        for (int i = 0; i < N; ++i) {
            int start = i * 2 * L;

            blockMergeLarge<NumThreads, K, V, L, Dir, Comp, FullMerge>(
                    listK + start, listV + start);
        }
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge = true>
inline __device__ void blockMerge(K* listK, V* listV) {
    constexpr bool kSmallerThanBlock = (L <= NumThreads);

    BlockMerge<
            NumThreads,
            K,
            V,
            N,
            L,
            Dir,
            Comp,
            kSmallerThanBlock,
            FullMerge>::merge(listK, listV);
}

// Specialization for block-wide monotonic merges producing a merge sort
// since what we really want is a constexpr loop expansion
template <
        int NumWarps,
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge {};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<1, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        // no merge required; single warp
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<2, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<4, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp>(sharedK, sharedV);
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 4),
                NumWarpQ * 2,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<8, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp>(sharedK, sharedV);
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 4),
                NumWarpQ * 2,
                !Dir,
                Comp>(sharedK, sharedV);
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 8),
                NumWarpQ * 4,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <
        typename K,
        typename V,
        bool Dir,
        typename Comp,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
struct BlockSelect {
    static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
    static constexpr int kTotalWarpSortSize = NumWarpQ;

    __device__ inline BlockSelect(
            K initKVal,
            V initVVal,
            K* smemK,
            V* smemV,
            int k)
            : initK(initKVal),
              initV(initVVal),
              numVals(0),
              warpKTop(initKVal),
              sharedK(smemK),
              sharedV(smemV),
              kMinus1(k - 1) {
        static_assert(
                isPowerOf2(ThreadsPerBlock),
                "threads must be a power-of-2");
        static_assert(
                isPowerOf2(NumWarpQ), "warp queue must be power-of-2");

        // Fill the per-thread queue keys with the default value
#pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            // threadV[i] = initV;
        }

        int laneId = getLaneId();
        int warpId = threadIdx.x / kWarpSize;
        warpK = sharedK + warpId * kTotalWarpSortSize;
        // warpV = sharedV + warpId * kTotalWarpSortSize;

        // Fill warp queue (only the actual queue space is fine, not where
        // we write the per-thread queues for merging)
        for (int i = laneId; i < NumWarpQ; i += kWarpSize) {
            warpK[i] = initK;
            // warpV[i] = initV;
        }

        warpFence();
    }

    __device__ inline void addThreadQ(K k, V v) {
        if (Dir ? Comp::gt(k, warpKTop) : Comp::lt(k, warpKTop)) {
            // Rotate right
#pragma unroll
            for (int i = NumThreadQ - 1; i > 0; --i) {
                threadK[i] = threadK[i - 1];
                // threadV[i] = threadV[i - 1];
            }

            threadK[0] = k;
            // threadV[0] = v;
            ++numVals;
        }
    }

    __device__ inline void checkThreadQ() {
        bool needSort = (numVals == NumThreadQ);

#if CUDA_VERSION >= 9000
        needSort = __any_sync(0xffffffff, needSort);
#else
        needSort = __any(needSort);
#endif

        if (!needSort) {
            // no lanes have triggered a sort
            return;
        }

        // This has a trailing warpFence
        mergeWarpQ();

        // Any top-k elements have been merged into the warp queue; we're
        // free to reset the thread queues
        numVals = 0;

#pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            // threadV[i] = initV;
        }

        // We have to beat at least this element
        warpKTop = warpK[kMinus1];

        warpFence();
    }

    /// This function handles sorting and merging together the
    /// per-thread queues with the warp-wide queue, creating a sorted
    /// list across both
    __device__ inline void mergeWarpQ() {
        int laneId = getLaneId();

        // Sort all of the per-thread queues
        warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

        constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;
        K warpKRegisters[kNumWarpQRegisters];
        V warpVRegisters[kNumWarpQRegisters];

#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpKRegisters[i] = warpK[i * kWarpSize + laneId];
            // warpVRegisters[i] = warpV[i * kWarpSize + laneId];
        }

        warpFence();

        // The warp queue is already sorted, and now that we've sorted the
        // per-thread queue, merge both sorted lists together, producing
        // one sorted list
        warpMergeAnyRegisters<
                K,
                V,
                kNumWarpQRegisters,
                NumThreadQ,
                !Dir,
                Comp,
                false>(warpKRegisters, warpVRegisters, threadK, threadV);

        // Write back out the warp queue
#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i * kWarpSize + laneId] = warpKRegisters[i];
            // warpV[i * kWarpSize + laneId] = warpVRegisters[i];
        }

        warpFence();
    }

    /// WARNING: all threads in a warp must participate in this.
    /// Otherwise, you must call the constituent parts separately.
    __device__ inline void add(K k, V v) {
        addThreadQ(k, v);
        checkThreadQ();
    }

    __device__ inline void reduce() {
        // Have all warps dump and merge their queues; this will produce
        // the final per-warp results
        mergeWarpQ();

        // block-wide dep; thus far, all warps have been completely
        // independent
        __syncthreads();

        // All warp queues are contiguous in smem.
        // Now, we have kNumWarps lists of NumWarpQ elements.
        // This is a power of 2.
        FinalBlockMerge<kNumWarps, ThreadsPerBlock, K, V, NumWarpQ, Dir, Comp>::
                merge(sharedK, sharedV);

        // The block-wide merge has a trailing syncthreads
    }

    // Default element key
    const K initK;

    // Default element value
    const V initV;

    // Number of valid elements in our thread queue
    int numVals;

    // The k-th highest (Dir) or lowest (!Dir) element
    K warpKTop;

    // Thread queue values
    K threadK[NumThreadQ];
    V threadV[NumThreadQ];

    // Queues for all warps
    K* sharedK;
    V* sharedV;

    // Our warp's queue (points into sharedK/sharedV)
    // warpK[0] is highest (Dir) or lowest (!Dir)
    K* warpK;
    V* warpV;

    // This is a cached k-1 value
    int kMinus1;
};

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
    return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
    return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
    return divUp(a, b) * b;
}

template <typename T>
struct Comparator {
    __device__ static inline bool lt(T a, T b) {
        return a < b;
    }

    __device__ static inline bool gt(T a, T b) {
        return a > b;
    }
};

template <
        typename K,
        typename IndexType,
        bool Dir,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void blockSelect(
        K *in, int ih, int iw,
        K *outK, int oh, int ow,
        K initK,
        IndexType initV,
        int k) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            K,
            IndexType,
            Dir,
            Comparator<K>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    idx_t row = blockIdx.x;

    idx_t i = threadIdx.x;
    K* inStart = in + row*iw + i; // [row][i];

    // Whole warps must participate in the selection
    idx_t limit = roundDown(iw, kWarpSize);

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inStart, (IndexType)i);
        inStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < iw) {
        heap.addThreadQ(*inStart, (IndexType)i);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < ow; i += ThreadsPerBlock) {
        if (i<k){
        *(outK + row*ow + i)= smemK[i]; // [row][i] 
        }
        else{
        *(outK + row*ow + i)= 0; // [row][i] 
        }
    }
}

#define BLOCK_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                         \
    void runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(                          \
            TYPE * in,   int ih, int iw,                                      \
            TYPE * outK, int oh, int ow,                                      \
            bool dir,                                                          \
            int k,                                                             \
            cudaStream_t stream) {                                             \
                                                                               \
        auto grid = dim3(ih);                                       \
                                                                               \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;    \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelect<                                                           \
                TYPE,                                                          \
                idx_t,                                                         \
                DIR,                                                           \
                WARP_Q,                                                        \
                THREAD_Q,                                                      \
                kBlockSelectNumThreads>                                        \
                <<<grid, block, 0, stream>>>(                                  \
                    in,     ih, iw,                                            \
                    outK,   oh, ow,                                            \
                    kInit, vInit, k);                                          \
        CUDA_TEST_ERROR();                                                     \
    }

BLOCK_SELECT_IMPL(uint64_t, true, 128, 3);

#define BLOCK_SELECT_CALL(TYPE, DIR, WARP_Q) \
    runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(in, ih, iw, outK, oh, ow, dir, k, stream)

void runBlockSelect(
        uint64_t *in,    int ih, int iw,
        uint64_t *outK,  int oh, int ow,
        bool dir,
        int k,
        cudaStream_t stream) {

    BLOCK_SELECT_CALL(uint64_t, true, 128);
}
} // namespace gpu
} // namespace faiss
