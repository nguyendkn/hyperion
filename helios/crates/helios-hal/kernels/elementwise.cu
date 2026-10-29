// Small bf16 kernels for the Qwen2 forward pass. Each mirrors the rounding torch performs
// after every bf16 operation (compute in f32, round to bf16), so results can be compared
// bit for bit with the CPU reference in src/reference.rs. Rebuild the PTX with
// scripts/build-ptx.sh.
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

static __device__ __forceinline__ float rd(bf16 v) { return __bfloat162float(v); }
static __device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16(v)); }

// out[t, :] = table[ids[t], :]. One block per token; the host guarantees ids < vocab.
extern "C" __global__ void embedding_gather_bf16(const unsigned int* __restrict__ ids,
                                                 const bf16* __restrict__ table,
                                                 bf16* __restrict__ out, int hidden) {
    const size_t t = blockIdx.x;
    const bf16* src = table + (size_t)ids[t] * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[t * hidden + i] = src[i];
}

// y = a + b
extern "C" __global__ void add_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                    bf16* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(rd(a[i]) + rd(b[i]));
}

// y = bf16(silu(gate)) * up, silu computed in f32 and rounded before the product.
extern "C" __global__ void swiglu_bf16(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                                       bf16* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = rd(gate[i]);
        const float silu = rnd(g / (1.f + expf(-g)));
        y[i] = __float2bfloat16(silu * rd(up[i]));
    }
}

// Rotary embedding, Hugging Face rotate_half layout, x is [tokens, heads, head_dim]. cos/sin
// are precomputed [max_pos, head_dim] tables. One block per (token, head); threads loop over
// head_dim/2 pairs. Every product and the sum are rounded to bf16.
extern "C" __global__ void rope_bf16(const bf16* __restrict__ x, const bf16* __restrict__ cos_t,
                                     const bf16* __restrict__ sin_t,
                                     const unsigned int* __restrict__ positions,
                                     bf16* __restrict__ y, int heads, int head_dim) {
    const int t = blockIdx.x / heads;
    const size_t base = (size_t)blockIdx.x * head_dim;
    const size_t table = (size_t)positions[t] * head_dim;
    const int half = head_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float x1 = rd(x[base + i]), x2 = rd(x[base + i + half]);
        const float c = rd(cos_t[table + i]), s = rd(sin_t[table + i]);
        y[base + i] = __float2bfloat16(rnd(x1 * c) + rnd(-x2 * s));
        y[base + i + half] = __float2bfloat16(rnd(x2 * c) + rnd(x1 * s));
    }
}

// Causal grouped-query attention for one (query token, head) per block. q is
// [tokens, heads, head_dim]; k and v are the cache [seq, kv_heads, head_dim]. Query t sits at
// absolute position q_start + t and attends to keys 0..=q_start + t, so the same kernel serves
// prefill (q_start = 0) and decode (one token at q_start = seq - 1). Scores, softmax and the
// weighted sum are accumulated in f32 and the output is rounded to bf16 once.
// Dynamic shared memory: (seq + blockDim.x) floats. Not yet tiled: seq is limited by it.
extern "C" __global__ void attention_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                          const bf16* __restrict__ v, bf16* __restrict__ out,
                                          int seq, int heads, int kv_heads, int head_dim, int q_start) {
    extern __shared__ float shared[];
    float* scores = shared;     // [seq]
    float* red = shared + seq;  // [blockDim.x]
    const int t = blockIdx.x / heads;
    const int h = blockIdx.x % heads;
    const int kvh = h / (heads / kv_heads);
    const int last = q_start + t;  // inclusive
    const bf16* qrow = q + (size_t)blockIdx.x * head_dim;
    const float scale = 1.f / sqrtf((float)head_dim);

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        const bf16* krow = k + ((size_t)j * kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += rd(qrow[d]) * rd(krow[d]);
        const float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= last; ++j) {
            acc += scores[j] * rd(v[((size_t)j * kv_heads + kvh) * head_dim + d]);
        }
        out[(size_t)blockIdx.x * head_dim + d] = __float2bfloat16(acc / total);
    }
}

// Append this step's K and V rows to the cache at row *pos (position read from device memory
// for the same reason as above).
extern "C" __global__ void kv_append_bf16(bf16* __restrict__ k_cache, bf16* __restrict__ v_cache,
                                          const bf16* __restrict__ k, const bf16* __restrict__ v,
                                          const int* __restrict__ pos, int kv_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    const size_t at = (size_t)(*pos) * kv_dim + i;
    k_cache[at] = k[i];
    v_cache[at] = v[i];
}

// out[i] = bf16(y[i] + bias[i % width]) where y is the f32 output of a matrix multiply. A null
// `bias` means no bias. This is torch's bf16 `Linear`: f32 accumulation, bias added in f32, one
// rounding to bf16.
extern "C" __global__ void bias_cast_bf16(const float* __restrict__ y, const bf16* __restrict__ bias,
                                          bf16* __restrict__ out, int n, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = y[i];
        if (bias != nullptr) v += rd(bias[i % width]);
        out[i] = __float2bfloat16(v);
    }
}

// Greedy token choice on the device, and the chaining of decode steps. Every f32 logit is first
// rounded to bf16 (the precision logits are defined at, and what the host path compares), the
// highest wins, the lowest index on ties, and a value that is NaN never wins. The winner is then
// written as the next step's input id, the position advances by one, and the token is stored at
// ring[position - ring_base] for the host to read later. One block of 1024 threads.
extern "C" __global__ void argmax_chain_bf16(const float* __restrict__ logits, int n, unsigned int* ids,
                                             int* pos, unsigned int* ring, int ring_base) {
    __shared__ float warp_value[32];
    __shared__ int warp_index[32];
    float best = -INFINITY;
    int best_at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = rnd(logits[i]);
        if (v > best) {
            best = v;
            best_at = i;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, best, offset);
        const int other_at = __shfl_down_sync(0xffffffffu, best_at, offset);
        if (other > best || (other == best && other_at < best_at)) {
            best = other;
            best_at = other_at;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        warp_value[threadIdx.x >> 5] = best;
        warp_index[threadIdx.x >> 5] = best_at;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        best = threadIdx.x < (blockDim.x >> 5) ? warp_value[threadIdx.x] : -INFINITY;
        best_at = threadIdx.x < (blockDim.x >> 5) ? warp_index[threadIdx.x] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(0xffffffffu, best, offset);
            const int other_at = __shfl_down_sync(0xffffffffu, best_at, offset);
            if (other > best || (other == best && other_at < best_at)) {
                best = other;
                best_at = other_at;
            }
        }
        if (threadIdx.x == 0) {
            const unsigned int token = best_at == 0x7fffffff ? 0u : (unsigned int)best_at;
            const int p = *pos;
            ring[p - ring_base] = token;
            ids[0] = token;
            *pos = p + 1;
        }
    }
}

// ---- Decode attention, split over the context ---------------------------------------------
// One query token (position *pos) attends to keys 0..=*pos. The work is cut into `splits`
// pieces of at least SPLIT_MIN_CHUNK positions, one block per (KV head, split), so the grid does
// not depend on the context length and a captured graph stays valid for every step; blocks without
// positions return at once. Shared memory is fixed too: positions go through in tiles of
// SPLIT_TILE with an online softmax. A block reads each K and V row once, with 16-byte loads, and
// serves every query head that shares the KV head. Each split writes (max, sum of exp,
// unnormalised weighted V) to the workspace; attention_combine_bf16 merges them in split order, so
// results are identical from run to run (no atomics). head_dim is a template parameter so the
// row loops unroll and the loads can be issued back to back.
#define SPLIT_TILE 128
// A split gets at least this many positions, so a short context uses only as many blocks as it has
// work for and the combine kernel merges only those.
#define SPLIT_MIN_CHUNK 64
#define MAX_GROUP 8  // query heads per KV head

// Eight bf16 values of a 16-byte load as floats.
static __device__ __forceinline__ void unpack8(const uint4 raw, float out[8]) {
    const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pair[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

template <int HD>
static __device__ __forceinline__ void attention_split_impl(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                            const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                            const int* __restrict__ pos, int heads, int kv_heads,
                                                            int splits) {
    constexpr int NV = HD / 8;            // 16-byte pieces per row
    constexpr int TPR = HD / 8;           // threads that together read one V row
    constexpr int ROWS = SPLIT_TILE / TPR;  // V rows read at the same time by the block
    __shared__ float q_s[MAX_GROUP][HD];
    __shared__ float sc[MAX_GROUP][SPLIT_TILE];
    __shared__ float part[4][MAX_GROUP][HD];
    __shared__ float wmax[MAX_GROUP][SPLIT_TILE / 32];
    __shared__ float wsum[MAX_GROUP][SPLIT_TILE / 32];
    __shared__ float m_run[MAX_GROUP];
    __shared__ float l_run[MAX_GROUP];

    const int g = blockIdx.x / splits;
    const int s = blockIdx.x % splits;
    const int group = heads / kv_heads;
    const int len = *pos + 1;
    const int chunk = max(SPLIT_MIN_CHUNK, (len + splits - 1) / splits);
    if (s * chunk >= len) return;  // no positions for this split (uniform across the block)
    const int lo = s * chunk;
    const int hi = min(len, lo + chunk);
    const float scale = 1.f / sqrtf((float)HD);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = threadIdx.x % TPR;  // which 8 dimensions of a V row this thread owns
    const int row = threadIdx.x / TPR;  // which of the rows read together

    for (int i = threadIdx.x; i < group * HD; i += SPLIT_TILE) q_s[i / HD][i % HD] = rd(q[(size_t)(g * group + i / HD) * HD + i % HD]);
    if (threadIdx.x < group) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.f;
    }
    float acc[MAX_GROUP][8];
#pragma unroll
    for (int h = 0; h < MAX_GROUP; ++h)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[h][e] = 0.f;
    __syncthreads();

    for (int base = lo; base < hi; base += SPLIT_TILE) {
        const int n = min(SPLIT_TILE, hi - base);
        // Scores of this thread's position for every query head of the group.
        float dots[MAX_GROUP];
#pragma unroll
        for (int h = 0; h < MAX_GROUP; ++h) dots[h] = 0.f;
        if (threadIdx.x < n) {
            const uint4* krow = reinterpret_cast<const uint4*>(k_cache + ((size_t)(base + threadIdx.x) * kv_heads + g) * HD);
            uint4 raw[NV];
#pragma unroll
            for (int v = 0; v < NV; ++v) raw[v] = krow[v];
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                float f[8];
                unpack8(raw[v], f);
#pragma unroll
                for (int h = 0; h < MAX_GROUP; ++h)
                    if (h < group)
#pragma unroll
                        for (int e = 0; e < 8; ++e) dots[h] += q_s[h][v * 8 + e] * f[e];
            }
        }
#pragma unroll
        for (int h = 0; h < MAX_GROUP; ++h) {
            if (h >= group) continue;
            const float sv = threadIdx.x < n ? dots[h] * scale : -INFINITY;
            sc[h][threadIdx.x] = sv;
            float w = sv;
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, off));
            if (lane == 0) wmax[h][warp] = w;
        }
        __syncthreads();
        float corr[MAX_GROUP];
#pragma unroll
        for (int h = 0; h < MAX_GROUP; ++h) {
            corr[h] = 1.f;
            if (h >= group) continue;
            float tile_max = wmax[h][0];
            for (int w = 1; w < SPLIT_TILE / 32; ++w) tile_max = fmaxf(tile_max, wmax[h][w]);
            const float m_new = fmaxf(m_run[h], tile_max);
            corr[h] = expf(m_run[h] - m_new);  // 0 on the first tile (m_run is -inf)
            const float p = threadIdx.x < n ? expf(sc[h][threadIdx.x] - m_new) : 0.f;
            sc[h][threadIdx.x] = p;
            float w = p;
            for (int off = 16; off > 0; off >>= 1) w += __shfl_xor_sync(0xffffffffu, w, off);
            if (lane == 0) wsum[h][warp] = w;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[h][e] *= corr[h];
        }
        __syncthreads();
        if (threadIdx.x < group) {
            const int h = threadIdx.x;
            float tile_max = wmax[h][0];
            float tile_sum = wsum[h][0];
            for (int w = 1; w < SPLIT_TILE / 32; ++w) {
                tile_max = fmaxf(tile_max, wmax[h][w]);
                tile_sum += wsum[h][w];
            }
            l_run[h] = l_run[h] * corr[h] + tile_sum;
            m_run[h] = fmaxf(m_run[h], tile_max);
        }
        // Weighted sum of V: each thread owns 8 dimensions of every ROWS-th position.
        for (int j = row; j < n; j += ROWS) {
            float f[8];
            unpack8(*reinterpret_cast<const uint4*>(v_cache + ((size_t)(base + j) * kv_heads + g) * HD + col * 8), f);
#pragma unroll
            for (int h = 0; h < MAX_GROUP; ++h)
                if (h < group) {
                    const float p = sc[h][j];
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[h][e] += p * f[e];
                }
        }
        __syncthreads();
    }

    // Add up the threads that read different rows: first inside the warp (fixed xor order),
    // then across the four warps in warp order.
#pragma unroll
    for (int h = 0; h < MAX_GROUP; ++h)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            for (int off = TPR; off < 32; off <<= 1) acc[h][e] += __shfl_xor_sync(0xffffffffu, acc[h][e], off);
    if (lane < TPR) {
#pragma unroll
        for (int h = 0; h < MAX_GROUP; ++h)
            if (h < group)
#pragma unroll
                for (int e = 0; e < 8; ++e) part[warp][h][col * 8 + e] = acc[h][e];
    }
    __syncthreads();
    const int stride = HD + 2;
    for (int i = threadIdx.x; i < group * HD; i += SPLIT_TILE) {
        const int h = i / HD, d = i % HD;
        float* dst = ws + ((size_t)(g * group + h) * splits + s) * stride;
        dst[2 + d] = ((part[0][h][d] + part[1][h][d]) + part[2][h][d]) + part[3][h][d];
        if (d == 0) {
            dst[0] = m_run[h];
            dst[1] = l_run[h];
        }
    }
}

// One entry point per head dimension, so each kernel's static shared memory is only its own.
extern "C" __global__ void attention_split_32_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                    const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                    const int* __restrict__ pos, int heads, int kv_heads, int splits) {
    attention_split_impl<32>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits);
}

extern "C" __global__ void attention_split_64_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                    const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                    const int* __restrict__ pos, int heads, int kv_heads, int splits) {
    attention_split_impl<64>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits);
}

extern "C" __global__ void attention_split_128_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                    const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                    const int* __restrict__ pos, int heads, int kv_heads, int splits) {
    attention_split_impl<128>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits);
}

// Merge the splits of every query head: out = sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s, in
// split order. One block per query head, one thread per dimension; the per-split weights are
// computed once per block, in parallel, and shared.
extern "C" __global__ void attention_combine_bf16(const float* __restrict__ ws, bf16* __restrict__ out,
                                                  const int* __restrict__ pos, int heads, int head_dim,
                                                  int splits) {
    __shared__ float split_max[256];
    __shared__ float split_weight[256];
    const int len = *pos + 1;
    const int chunk = max(SPLIT_MIN_CHUNK, (len + splits - 1) / splits);
    const int active = (len + chunk - 1) / chunk;  // splits that had work, the only ones that wrote
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int stride = head_dim + 2;
    const float* base = ws + (size_t)h * splits * stride;
    for (int s = threadIdx.x; s < active; s += blockDim.x) split_max[s] = base[(size_t)s * stride];
    __syncthreads();
    float top = -INFINITY;
    for (int s = 0; s < active; ++s) top = fmaxf(top, split_max[s]);
    for (int s = threadIdx.x; s < active; s += blockDim.x) split_weight[s] = expf(split_max[s] - top);
    __syncthreads();
    float total = 0.f, acc = 0.f;
#pragma unroll 4
    for (int s = 0; s < active; ++s) {
        const float w = split_weight[s];
        total += base[(size_t)s * stride + 1] * w;
        acc += base[(size_t)s * stride + 2 + d] * w;
    }
    out[(size_t)h * head_dim + d] = __float2bfloat16(acc / total);
}
