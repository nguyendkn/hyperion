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
