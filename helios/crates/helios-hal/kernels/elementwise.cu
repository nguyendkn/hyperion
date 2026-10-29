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
// pieces of ceil(len / splits) positions, one block per (KV head, split), so the grid does not
// depend on the context length and a captured graph stays valid for every step; shared memory
// is fixed too (positions are processed in tiles of SPLIT_TILE with an online softmax). A block
// reads each K and V row once and serves every query head that shares the KV head. Each split
// writes (max, sum of exp, unnormalised weighted V) to the workspace; attention_combine_bf16 merges
// them in split order, so results are identical from run to run (no atomics).
#define SPLIT_TILE 128
#define MAX_GROUP 8  // query heads per KV head

extern "C" __global__ void attention_split_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                const bf16* __restrict__ v_cache, float* __restrict__ ws,
                                                const int* __restrict__ pos, int heads, int kv_heads,
                                                int head_dim, int splits) {
    __shared__ float q_s[MAX_GROUP][128];
    __shared__ float sc[MAX_GROUP][SPLIT_TILE];
    __shared__ float part[MAX_GROUP][128];
    __shared__ float wmax[MAX_GROUP][SPLIT_TILE / 32];
    __shared__ float wsum[MAX_GROUP][SPLIT_TILE / 32];
    __shared__ float m_run[MAX_GROUP];
    __shared__ float l_run[MAX_GROUP];

    const int g = blockIdx.x / splits;
    const int s = blockIdx.x % splits;
    const int group = heads / kv_heads;
    const int len = *pos + 1;
    const int chunk = (len + splits - 1) / splits;
    const int lo = s * chunk;
    const int hi = min(len, lo + chunk);
    const float scale = 1.f / sqrtf((float)head_dim);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int d = threadIdx.x % head_dim;       // this thread's output dimension
    const int jg = threadIdx.x / head_dim;      // which interleaved share of the tile's positions
    const int jgroups = blockDim.x / head_dim;

    for (int i = threadIdx.x; i < group * head_dim; i += blockDim.x)
        q_s[i / head_dim][i % head_dim] = rd(q[(size_t)(g * group + i / head_dim) * head_dim + i % head_dim]);
    if (threadIdx.x < group) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.f;
    }
    float acc[MAX_GROUP];
    for (int h = 0; h < MAX_GROUP; ++h) acc[h] = 0.f;
    __syncthreads();

    for (int base = lo; base < hi; base += SPLIT_TILE) {
        const int n = min(SPLIT_TILE, hi - base);
        // Scores of this thread's position for every query head of the group.
        float dots[MAX_GROUP];
        for (int h = 0; h < MAX_GROUP; ++h) dots[h] = 0.f;
        if (threadIdx.x < n) {
            const bf16* krow = k_cache + ((size_t)(base + threadIdx.x) * kv_heads + g) * head_dim;
            for (int dd = 0; dd < head_dim; ++dd) {
                const float kv = rd(krow[dd]);
                for (int h = 0; h < group; ++h) dots[h] += q_s[h][dd] * kv;
            }
        }
        for (int h = 0; h < group; ++h) {
            const float sv = threadIdx.x < n ? dots[h] * scale : -INFINITY;
            sc[h][threadIdx.x] = sv;
            float w = sv;
            for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, off));
            if (lane == 0) wmax[h][warp] = w;
        }
        __syncthreads();
        float corr[MAX_GROUP];
        for (int h = 0; h < group; ++h) {
            float tile_max = wmax[h][0];
            for (int w = 1; w < SPLIT_TILE / 32; ++w) tile_max = fmaxf(tile_max, wmax[h][w]);
            const float m_new = fmaxf(m_run[h], tile_max);
            corr[h] = expf(m_run[h] - m_new);  // 0 on the first tile (m_run is -inf)
            const float p = threadIdx.x < n ? expf(sc[h][threadIdx.x] - m_new) : 0.f;
            sc[h][threadIdx.x] = p;
            float w = p;
            for (int off = 16; off > 0; off >>= 1) w += __shfl_xor_sync(0xffffffffu, w, off);
            if (lane == 0) wsum[h][warp] = w;
            acc[h] *= corr[h];
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
        // Weighted sum of V: each thread owns one dimension and every jgroups-th position.
        for (int j = jg; j < n; j += jgroups) {
            const float vv = rd(v_cache[((size_t)(base + j) * kv_heads + g) * head_dim + d]);
            for (int h = 0; h < group; ++h) acc[h] += sc[h][j] * vv;
        }
        __syncthreads();
    }

    for (int h = 0; h < group; ++h) part[h][jg * head_dim + d] = acc[h];
    __syncthreads();
    const int stride = head_dim + 2;
    for (int h = 0; h < group; ++h) {
        float* dst = ws + ((size_t)(g * group + h) * splits + s) * stride;
        for (int dd = threadIdx.x; dd < head_dim; dd += blockDim.x) {
            float total = 0.f;
            for (int j = 0; j < jgroups; ++j) total += part[h][j * head_dim + dd];
            dst[2 + dd] = total;
        }
        if (threadIdx.x == 0) {
            dst[0] = m_run[h];
            dst[1] = l_run[h];
        }
    }
}

// Merge the splits of every query head: out = sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s, in
// split order. One block per query head, one thread per dimension.
extern "C" __global__ void attention_combine_bf16(const float* __restrict__ ws, bf16* __restrict__ out,
                                                  int heads, int head_dim, int splits) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int stride = head_dim + 2;
    const float* base = ws + (size_t)h * splits * stride;
    float top = -INFINITY;
    for (int s = 0; s < splits; ++s) top = fmaxf(top, base[(size_t)s * stride]);
    float total = 0.f, acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float m = base[(size_t)s * stride];
        if (m == -INFINITY) continue;  // an empty split
        const float w = expf(m - top);
        total += base[(size_t)s * stride + 1] * w;
        acc += base[(size_t)s * stride + 2 + d] * w;
    }
    out[(size_t)h * head_dim + d] = __float2bfloat16(acc / total);
}
