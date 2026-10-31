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

// Greedy token choice on the device, and the chaining of decode steps. The highest f32 logit wins
// (as the output head computed it, not rounded to bf16: the host argmax over the same f32 values picks the same
// token), the lowest index on ties, and a value that is NaN never wins. The winner is then
// written as the next step's input id, the position advances by one, and the token is stored at
// ring[position - *ring_base] for the host to read later. `ring_base` is read from device memory, like
// the position, so a graph captured once at load time serves every generation. One block of 1024 threads.
extern "C" __global__ void argmax_chain_bf16(const float* __restrict__ logits, int n, unsigned int* ids,
                                             int* pos, unsigned int* ring, const int* ring_base) {
    __shared__ float warp_value[32];
    __shared__ int warp_index[32];
    float best = -INFINITY;
    int best_at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
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
            ring[p - *ring_base] = token;
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

// Where the K (which = 0) or V (which = 1) row of position j and KV head g lives. LAYOUT 0: one contiguous [position][kv_head][head_dim]
// buffer per layer (what the decode step uses today). LAYOUT 1 (layer-major pages): the same, cut into pages of 64 positions placed by
// `table[position / 64]` inside this layer's K (or V) pool. LAYOUT 2 (page-major): one pool for every layer, a page holding
// [layer][k|v][token][kv_head][head_dim], `base` being the pool for K and V alike. These variants exist to measure the cost of the page
// layouts of ADR 0002 against each other; the arithmetic and its order are identical in all three.
template <int HD, int LAYOUT>
static __device__ __forceinline__ const bf16* kv_row(const bf16* base, const int* table, int layer, int layers, int kv_heads, int j, int g, int which) {
    if (LAYOUT == 0) return base + ((size_t)j * kv_heads + g) * HD;
    const int page = table[j >> 6], tok = j & 63;
    if (LAYOUT == 1) return base + (((size_t)page * 64 + tok) * kv_heads + g) * HD;
    return base + ((((size_t)page * layers + layer) * 2 + which) * 64 + tok) * (size_t)kv_heads * HD + (size_t)g * HD;
}

template <int HD, int LAYOUT>
static __device__ __forceinline__ void attention_split_impl(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                            const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                            const int* __restrict__ pos, int heads, int kv_heads,
                                                            int splits, const int* __restrict__ table, int layer, int layers) {
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

    // A table whose first entry is NO_PAGE (-1) holds nothing: the warm-up launches that run before the pool is mapped skip every access.
    if (LAYOUT != 0 && table[0] < 0) return;
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
            const uint4* krow = reinterpret_cast<const uint4*>(kv_row<HD, LAYOUT>(k_cache, table, layer, layers, kv_heads, base + threadIdx.x, g, 0));
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
            unpack8(*reinterpret_cast<const uint4*>(kv_row<HD, LAYOUT>(v_cache, table, layer, layers, kv_heads, base + j, g, 1) + col * 8), f);
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
extern "C" __global__ void attention_split_64_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                    const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                    const int* __restrict__ pos, int heads, int kv_heads, int splits) {
    attention_split_impl<64, 0>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits, nullptr, 0, 0);
}

extern "C" __global__ void attention_split_128_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                    const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                    const int* __restrict__ pos, int heads, int kv_heads, int splits) {
    attention_split_impl<128, 0>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits, nullptr, 0, 0);
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

// ---- Chunked prefill -------------------------------------------------------------------------
// A prompt is processed in chunks of a fixed capacity (a power of two, 128 to 4096), one captured graph
// per capacity. A chunk with fewer real rows than its capacity is padded: its padding rows run through
// the same kernels, so what makes a graph valid for every chunk is that nothing here depends on a launch
// argument that changes. Row t sits at absolute position positions[t]; *rows is the number of real rows.
// Padding rows carry the last real position (a valid table row and cache row to read) and are never
// written to the cache.

// Append the chunk's K and V rows to the cache: row t goes to cache position positions[t]. One block per
// row, threads loop over kv_dim. Rows at or beyond *rows are padding and write nothing.
extern "C" __global__ void kv_append_chunk_bf16(bf16* __restrict__ k_cache, bf16* __restrict__ v_cache,
                                                const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                const unsigned int* __restrict__ positions,
                                                const int* __restrict__ rows, int kv_dim) {
    const int t = blockIdx.x;
    if (t >= *rows) return;
    const size_t at = (size_t)positions[t] * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        k_cache[at + i] = k[(size_t)t * kv_dim + i];
        v_cache[at + i] = v[(size_t)t * kv_dim + i];
    }
}

// Block-wide max / sum over exactly 128 threads (four warps). `red` has room for 4 floats. Every thread
// gets the result; the trailing barrier lets the same buffer be reused at once.
static __device__ __forceinline__ float block_max_128(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float r = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    return r;
}

static __device__ __forceinline__ float block_sum_128(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float r = (red[0] + red[1]) + (red[2] + red[3]);
    __syncthreads();
    return r;
}

// Causal grouped-query attention of one chunk row for one head per block of 128 threads, over keys
// 0..=positions[t], with an online softmax over tiles of 128 keys: shared memory is head_dim + 132
// floats whatever the context length, which is what lifts the 8192 limit of attention_bf16. Scores,
// softmax statistics and the weighted sum are f32; the output is rounded to bf16 once. head_dim must be a
// multiple of 8 and at most 256 (the host admits 64 and 128). q is [rows, heads, head_dim]; the caches
// are [positions, kv_heads, head_dim].
extern "C" __global__ void attention_chunk_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                const bf16* __restrict__ v_cache, bf16* __restrict__ out,
                                                const unsigned int* __restrict__ positions,
                                                int heads, int kv_heads, int head_dim) {
    extern __shared__ float shared[];
    float* qs = shared;                // [head_dim]
    float* p = shared + head_dim;      // [128]: scores, then weights, of the current tile
    float* red = p + 128;              // [4]
    const int t = blockIdx.x / heads;
    const int h = blockIdx.x % heads;
    const int kvh = h / (heads / kv_heads);
    const int last = (int)positions[t];  // inclusive
    const size_t row_stride = (size_t)kv_heads * head_dim;
    const bf16* qrow = q + (size_t)blockIdx.x * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) qs[d] = rd(qrow[d]);
    __syncthreads();
    const float scale = 1.f / sqrtf((float)head_dim);

    float m = -INFINITY, l = 0.f;
    float acc0 = 0.f, acc1 = 0.f;  // output dimensions threadIdx.x and threadIdx.x + 128
    for (int base = 0; base <= last; base += 128) {
        const int j = base + threadIdx.x;
        float s = -INFINITY;
        if (j <= last) {
            const bf16* krow = k_cache + (size_t)j * row_stride + (size_t)kvh * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; d += 8) {
                const uint4 raw = *reinterpret_cast<const uint4*>(krow + d);
                const bf16* kp = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
                for (int e = 0; e < 8; ++e) dot += qs[d + e] * rd(kp[e]);
            }
            s = dot * scale;
        }
        const float tile_max = block_max_128(s, red);
        const float m_new = fmaxf(m, tile_max);  // finite: key `base` is always valid
        const float e = (j <= last) ? expf(s - m_new) : 0.f;
        p[threadIdx.x] = e;
        const float tile_sum = block_sum_128(e, red);  // also orders the writes to p
        const float correction = expf(m - m_new);      // 0 on the first tile (m = -inf)
        l = l * correction + tile_sum;
        acc0 *= correction;
        acc1 *= correction;
        const int count = min(128, last - base + 1);
        if (threadIdx.x < head_dim) {
            for (int i = 0; i < count; ++i) {
                acc0 += p[i] * rd(v_cache[(size_t)(base + i) * row_stride + (size_t)kvh * head_dim + threadIdx.x]);
            }
        }
        if (threadIdx.x + 128 < head_dim) {
            for (int i = 0; i < count; ++i) {
                acc1 += p[i] * rd(v_cache[(size_t)(base + i) * row_stride + (size_t)kvh * head_dim + threadIdx.x + 128]);
            }
        }
        __syncthreads();  // p is rewritten by the next tile
        m = m_new;
    }
    bf16* orow = out + (size_t)blockIdx.x * head_dim;
    if (threadIdx.x < head_dim) orow[threadIdx.x] = __float2bfloat16(acc0 / l);
    if (threadIdx.x + 128 < head_dim) orow[threadIdx.x + 128] = __float2bfloat16(acc1 / l);
}

// ---- Prefill attention on tensor cores (FlashAttention-2 style, mma.sync m16n8k16, sm80 and later) ----------------------------------
// Same job and the same inputs as attention_chunk_bf16 (row t of the chunk sits at positions[t] and attends to keys 0..=positions[t] of the
// cache), computed with bf16 tensor-core products and f32 accumulation. The work is cut over "virtual rows": virtual row v is query head
// (kvh * G + v % G) of chunk row v / G, G = heads / kv_heads, so one block owns 64 consecutive virtual rows of ONE kv head and loads every
// K and V tile once for all the query heads that share it. Four warps, 16 virtual rows each; keys run through in tiles of 64 with an online
// softmax; nothing is atomic, so a replay is bit-identical to the launch that was captured. head_dim 64 or 128.
//
// Fragment layouts of mma.m16n8k16 (g = lane / 4, q = lane % 4): A (16x16, row major): a0 = (row g, cols 2q..2q+1), a1 = (row g+8, same cols),
// a2 = (row g, cols 8+2q..), a3 = (row g+8, cols 8+2q..). B (16x8, "col"): b0 = (k 2q..2q+1, n g), b1 = (k 8+2q.., n g). C (16x8): c0,c1 =
// (row g, cols 2q..2q+1), c2,c3 = (row g+8, same cols).
static __device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

// PAGED: K and V are read through a device page table out of the page-major pool (`k_cache` and `v_cache` are then the same pointer); the
// arithmetic and its order are the same as for the contiguous cache.
template <int HD, bool PAGED>
static __device__ __forceinline__ void attention_mma_impl(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                          const bf16* __restrict__ v_cache, bf16* __restrict__ out,
                                                          const unsigned int* __restrict__ positions, int rows, int heads, int kv_heads,
                                                          const int* __restrict__ table, int layer, int layers) {
    if (PAGED && table[0] < 0) return;  // NO_PAGE: a warm-up launch before the pool is mapped
    constexpr int KT = HD / 16;   // k steps of the Q K^T product
    constexpr int NT = 8;         // n tiles of 8 keys in a 64-key tile
    constexpr int OT = HD / 8;    // n tiles of the output
    constexpr int LD = HD + 8;    // padded smem row (elements): keeps the fragment reads off one bank
    __shared__ __align__(16) bf16 ks[64 * LD];
    __shared__ __align__(16) bf16 vs[64 * LD];

    const int group = heads / kv_heads;
    const int kvh = blockIdx.y;
    const int total = rows * group;  // virtual rows of this kv head
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int v0 = blockIdx.x * 64 + warp * 16 + g;
    const int v1 = v0 + 8;
    const int pos0 = v0 < total ? (int)positions[v0 / group] : -1;
    const int pos1 = v1 < total ? (int)positions[v1 / group] : -1;
    const int vlast = min(blockIdx.x * 64 + 63, total - 1);
    const int kmax = (int)positions[vlast / group] + 1;  // keys any row of this block can see

    // Q fragments straight from global memory (each element is read once).
    unsigned qa[KT][4];
    {
        const bf16* r0 = q + ((size_t)(v0 / group) * heads + kvh * group + v0 % group) * HD;
        const bf16* r1 = q + ((size_t)(v1 / group) * heads + kvh * group + v1 % group) * HD;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            qa[kk][0] = v0 < total ? *reinterpret_cast<const unsigned*>(r0 + kk * 16 + 2 * qd) : 0u;
            qa[kk][1] = v1 < total ? *reinterpret_cast<const unsigned*>(r1 + kk * 16 + 2 * qd) : 0u;
            qa[kk][2] = v0 < total ? *reinterpret_cast<const unsigned*>(r0 + kk * 16 + 8 + 2 * qd) : 0u;
            qa[kk][3] = v1 < total ? *reinterpret_cast<const unsigned*>(r1 + kk * 16 + 8 + 2 * qd) : 0u;
        }
    }

    float o[OT][4];
#pragma unroll
    for (int i = 0; i < OT; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const float scale_log2 = 1.4426950408889634f * rsqrtf((float)HD);  // 1/sqrt(head_dim), in base 2

    for (int kb = 0; kb < kmax; kb += 64) {
        __syncthreads();  // the previous tile is no longer read
        for (int c = threadIdx.x; c < 64 * (HD / 8); c += 128) {
            const int r = c / (HD / 8), cc = c % (HD / 8);
            const int j = kb + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < kmax) {
                if (PAGED) {
                    const size_t page = (size_t)table[j >> 6];
                    const size_t row = (((page * layers + layer) * 2) * 64 + (j & 63)) * (size_t)kv_heads * HD + (size_t)kvh * HD + cc * 8;
                    kv = *reinterpret_cast<const uint4*>(k_cache + row);
                    vv = *reinterpret_cast<const uint4*>(v_cache + row + (size_t)64 * kv_heads * HD);
                } else {
                    const size_t at = ((size_t)j * kv_heads + kvh) * HD + cc * 8;
                    kv = *reinterpret_cast<const uint4*>(k_cache + at);
                    vv = *reinterpret_cast<const uint4*>(v_cache + at);
                }
            }
            *reinterpret_cast<uint4*>(ks + r * LD + cc * 8) = kv;
            *reinterpret_cast<uint4*>(vs + r * LD + cc * 8) = vv;
        }
        __syncthreads();

        float s[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const bf16* kr = ks + (nt * 8 + g) * LD + kk * 16 + 2 * qd;
                mma_16816(s[nt], qa[kk], *reinterpret_cast<const unsigned*>(kr), *reinterpret_cast<const unsigned*>(kr + 8));
            }
        }
        // Causal mask and scale (into the log2 domain).
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = kb + nt * 8 + 2 * qd + (e & 1);
                const int pos = e < 2 ? pos0 : pos1;
                s[nt][e] = col <= pos ? s[nt][e] * scale_log2 : -INFINITY;
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float u0 = n0 == -INFINITY ? 0.f : n0, u1 = n1 == -INFINITY ? 0.f : n1;  // a fully masked row keeps zero weight, no NaN
        const float c0 = exp2f(m0 - u0), c1 = exp2f(m1 - u1);                           // exp2f(-inf) = 0
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - u0);
            s[nt][1] = exp2f(s[nt][1] - u0);
            s[nt][2] = exp2f(s[nt][2] - u1);
            s[nt][3] = exp2f(s[nt][3] - u1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < OT; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }

        // O += P V, 16 keys at a time.
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            unsigned pa[4];
            pa[0] = pack_bf16x2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack_bf16x2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack_bf16x2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack_bf16x2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
            for (int ot = 0; ot < OT; ++ot) {
                const bf16* vc = vs + (kc * 16 + 2 * qd) * LD + ot * 8 + g;
                const unsigned b0 = pack_bf16x2(rd(vc[0]), rd(vc[LD]));
                const unsigned b1 = pack_bf16x2(rd(vc[8 * LD]), rd(vc[9 * LD]));
                mma_16816(o[ot], pa, b0, b1);
            }
        }
    }

    // Normalise and store: rows v0 (c0, c1) and v1 (c2, c3).
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    if (v0 < total) {
        bf16* orow = out + ((size_t)(v0 / group) * heads + kvh * group + v0 % group) * HD;
#pragma unroll
        for (int ot = 0; ot < OT; ++ot) *reinterpret_cast<unsigned*>(orow + ot * 8 + 2 * qd) = pack_bf16x2(o[ot][0] * i0, o[ot][1] * i0);
    }
    if (v1 < total) {
        bf16* orow = out + ((size_t)(v1 / group) * heads + kvh * group + v1 % group) * HD;
#pragma unroll
        for (int ot = 0; ot < OT; ++ot) *reinterpret_cast<unsigned*>(orow + ot * 8 + 2 * qd) = pack_bf16x2(o[ot][2] * i1, o[ot][3] * i1);
    }
}

// grid (ceil(rows * heads / kv_heads / 64), kv_heads), 128 threads, no dynamic shared memory.
extern "C" __global__ void attention_mma_64_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache,
                                                 bf16* __restrict__ out, const unsigned int* __restrict__ positions, int rows, int heads, int kv_heads) {
    attention_mma_impl<64, false>(q, k_cache, v_cache, out, positions, rows, heads, kv_heads, nullptr, 0, 0);
}

extern "C" __global__ void attention_mma_128_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache,
                                                  bf16* __restrict__ out, const unsigned int* __restrict__ positions, int rows, int heads, int kv_heads) {
    attention_mma_impl<128, false>(q, k_cache, v_cache, out, positions, rows, heads, kv_heads, nullptr, 0, 0);
}

// The prefill attention reading the page-major pool through the page table of the sequence (`pool` is both the K and the V pointer).
extern "C" __global__ void attention_mma_paged_64_bf16(const bf16* __restrict__ q, const bf16* __restrict__ pool, bf16* __restrict__ out,
                                                       const unsigned int* __restrict__ positions, int rows, int heads, int kv_heads,
                                                       const int* __restrict__ table, int layer, int layers) {
    attention_mma_impl<64, true>(q, pool, pool, out, positions, rows, heads, kv_heads, table, layer, layers);
}

extern "C" __global__ void attention_mma_paged_128_bf16(const bf16* __restrict__ q, const bf16* __restrict__ pool, bf16* __restrict__ out,
                                                        const unsigned int* __restrict__ positions, int rows, int heads, int kv_heads,
                                                        const int* __restrict__ table, int layer, int layers) {
    attention_mma_impl<128, true>(q, pool, pool, out, positions, rows, heads, kv_heads, table, layer, layers);
}

// Append this step's K and V (one token at *pos) to the page-major pool through the page table. The row of page p, layer l is
// [k|v][token][kv_head][head_dim]; a NO_PAGE entry (-1) is skipped (a warm-up launch before the pool is mapped).
extern "C" __global__ void kv_append_paged_bf16(bf16* __restrict__ pool, const int* __restrict__ table, const bf16* __restrict__ k,
                                                const bf16* __restrict__ v, const int* __restrict__ pos, int kv_dim, int layer, int layers) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    const int p = *pos;
    const int page = table[p >> 6];
    if (page < 0) return;
    const size_t base = (((size_t)page * layers + layer) * 2 * 64 + (p & 63)) * (size_t)kv_dim + i;
    pool[base] = k[i];
    pool[base + (size_t)64 * kv_dim] = v[i];
}

// The same for the rows of a prefill chunk: row t goes to position positions[t]; rows past *rows are padding and write nothing.
extern "C" __global__ void kv_append_chunk_paged_bf16(bf16* __restrict__ pool, const int* __restrict__ table, const bf16* __restrict__ k,
                                                      const bf16* __restrict__ v, const unsigned int* __restrict__ positions,
                                                      const int* __restrict__ rows, int kv_dim, int layer, int layers) {
    const int t = blockIdx.x;
    if (t >= *rows) return;
    const int p = (int)positions[t];
    const int page = table[p >> 6];
    if (page < 0) return;
    const size_t base = (((size_t)page * layers + layer) * 2 * 64 + (p & 63)) * (size_t)kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        pool[base + i] = k[(size_t)t * kv_dim + i];
        pool[base + (size_t)64 * kv_dim + i] = v[(size_t)t * kv_dim + i];
    }
}

// Page-table variants of the decode attention (see kv_row): the same kernel reading K and V through a device page table. For the layout
// benchmark of ADR 0002 and, in the page-major form, by the decode step.
#define PAGED_ENTRY(HD, LAYOUT, NAME)                                                                                         \
    extern "C" __global__ void NAME(const bf16* __restrict__ q, const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache, \
                                    float* __restrict__ ws, const int* __restrict__ pos, int heads, int kv_heads, int splits,    \
                                    const int* __restrict__ table, int layer, int layers) {                                      \
        attention_split_impl<HD, LAYOUT>(q, k_cache, v_cache, ws, pos, heads, kv_heads, splits, table, layer, layers);          \
    }
PAGED_ENTRY(64, 1, attention_split_paged_lm_64_bf16)
PAGED_ENTRY(128, 1, attention_split_paged_lm_128_bf16)
PAGED_ENTRY(64, 2, attention_split_paged_pm_64_bf16)
PAGED_ENTRY(128, 2, attention_split_paged_pm_128_bf16)

// ---- Batched decode over slots (page-major pool) -------------------------------------------------------------------------------
// A decode step of a bucket of B rows. Row r belongs to slot `slots[r]` (-1: an inactive row, masked: it reads nothing, writes no KV, produces
// no token and costs no attention work). The page tables of all slots are one array `tables[slot][stride]` of page indices (NO_PAGE = -1), so a
// row finds its table at `tables + slot * stride`. Per-slot device state: `slot_token[slot]` is the token the next step of that slot feeds
// in (the chaining of decode steps). Nothing here uses atomics: results are identical from run to run, and the result of a row does not depend
// on the other rows.

// ids[r] = the token slot `slots[r]` feeds in; an inactive row feeds token 0.
extern "C" __global__ void gather_inputs_rows(const int* __restrict__ slots, const unsigned int* __restrict__ slot_token,
                                              unsigned int* __restrict__ ids, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int s = slots[r];
    ids[r] = s >= 0 ? slot_token[s] : 0u;
}

// Apply page-table deltas: pairs[2i] is the word of the table array, pairs[2i + 1] the page index (RELEASED = u32::MAX = NO_PAGE = -1).
extern "C" __global__ void apply_page_deltas(int* __restrict__ tables, const unsigned int* __restrict__ pairs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tables[pairs[2 * i]] = (int)pairs[2 * i + 1];
}

// Cut the fused projection [rows][q_dim + 2 kv_dim] into q [rows][q_dim], k [rows][kv_dim], v [rows][kv_dim].
extern "C" __global__ void qkv_split_rows_bf16(const bf16* __restrict__ qkv, bf16* __restrict__ q, bf16* __restrict__ k,
                                               bf16* __restrict__ v, int q_dim, int kv_dim) {
    const int r = blockIdx.y;
    const int width = q_dim + 2 * kv_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x) {
        const bf16 value = qkv[(size_t)r * width + i];
        if (i < q_dim) q[(size_t)r * q_dim + i] = value;
        else if (i < q_dim + kv_dim) k[(size_t)r * kv_dim + (i - q_dim)] = value;
        else v[(size_t)r * kv_dim + (i - q_dim - kv_dim)] = value;
    }
}

// act[r][i] = bf16(silu(gu[r][i])) * gu[r][inter + i] for the fused gate|up projection [rows][2 inter]: the arithmetic of swiglu_bf16.
extern "C" __global__ void swiglu_rows_bf16(const bf16* __restrict__ gu, bf16* __restrict__ act, int inter) {
    const int r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = rd(gu[(size_t)r * 2 * inter + i]);
    const float silu = rnd(g / (1.f + expf(-g)));
    act[(size_t)r * inter + i] = __float2bfloat16(silu * rd(gu[(size_t)r * 2 * inter + inter + i]));
}

// Append row r K and V (one token at positions[r]) to the pool through the page table of its slot. Grid (ceil(kv_dim / blockDim), rows).
extern "C" __global__ void kv_append_rows_paged_bf16(bf16* __restrict__ pool, const int* __restrict__ tables, int stride,
                                                     const int* __restrict__ slots, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                     const int* __restrict__ positions, int kv_dim, int layer, int layers) {
    const int r = blockIdx.y;
    const int s = slots[r];
    if (s < 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    const int p = positions[r];
    const int page = tables[(size_t)s * stride + (p >> 6)];
    if (page < 0) return;
    const size_t base = (((size_t)page * layers + layer) * 2 * 64 + (p & 63)) * (size_t)kv_dim + i;
    pool[base] = k[(size_t)r * kv_dim + i];
    pool[base + (size_t)64 * kv_dim] = v[(size_t)r * kv_dim + i];
}

// Split-K decode attention of row blockIdx.y over the page-major pool: the kernel body of attention_split_*_paged_pm, one row per grid row.
// `ws` holds heads * splits * (head_dim + 2) floats per row.
#define ROWS_ENTRY(HD, NAME)                                                                                                           \
    extern "C" __global__ void NAME(const bf16* __restrict__ q, const bf16* __restrict__ pool, float* __restrict__ ws,                  \
                                    const int* __restrict__ positions, int heads, int kv_heads, int splits,                             \
                                    const int* __restrict__ tables, int stride, const int* __restrict__ slots, int layer, int layers) { \
        const int r = blockIdx.y;                                                                                                        \
        const int s = slots[r];                                                                                                          \
        if (s < 0) return;                                                                                                               \
        attention_split_impl<HD, 2>(q + (size_t)r * heads * HD, pool, pool, ws + (size_t)r * heads * splits * (HD + 2), positions + r,   \
                                    heads, kv_heads, splits, tables + (size_t)s * stride, layer, layers);                               \
    }
ROWS_ENTRY(64, attention_split_rows_64_bf16)
ROWS_ENTRY(128, attention_split_rows_128_bf16)

// Merge the splits of every (row, query head): grid (heads, rows), one thread per dimension; the arithmetic of attention_combine_bf16.
extern "C" __global__ void attention_combine_rows_bf16(const float* __restrict__ ws, bf16* __restrict__ out, const int* __restrict__ positions,
                                                       const int* __restrict__ slots, int heads, int head_dim, int splits) {
    __shared__ float split_max[256];
    __shared__ float split_weight[256];
    const int r = blockIdx.y;
    if (slots[r] < 0) return;
    const int len = positions[r] + 1;
    const int chunk = max(SPLIT_MIN_CHUNK, (len + splits - 1) / splits);
    const int active = (len + chunk - 1) / chunk;
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int stride = head_dim + 2;
    const float* base = ws + ((size_t)r * heads + h) * splits * stride;
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
    out[((size_t)r * heads + h) * head_dim + d] = __float2bfloat16(acc / total);
}

// Greedy token choice of every active row: block r takes the highest f32 logit of logits[r][0..n) (lowest index on ties, NaN never wins, as
// argmax_chain_bf16), stores it as slot_token[slot] (the next step input of that slot) and in the row ring tokens_ring[r][position & 63] (the host reads it at the
// position the token was sampled at, at most 63 steps late), and advances positions[r].
// An inactive row is skipped. One block of 1024 threads per row.
extern "C" __global__ void argmax_rows_bf16(const float* __restrict__ logits, int n, const int* __restrict__ slots, unsigned int* slot_token,
                                            unsigned int* tokens_ring, int* positions) {
    __shared__ float warp_value[32];
    __shared__ int warp_index[32];
    const int r = blockIdx.x;
    const int s = slots[r];
    if (s < 0) return;
    const float* row = logits + (size_t)r * n;
    float best = -INFINITY;
    int best_at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
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
            slot_token[s] = token;
            tokens_ring[(size_t)r * 64 + (positions[r] & 63)] = token;
            positions[r] += 1;
        }
    }
}
