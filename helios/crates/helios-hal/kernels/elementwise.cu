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
