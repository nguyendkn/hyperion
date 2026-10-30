// RMSNorm over bf16 rows, matching the Hugging Face Qwen2 formulation:
//   normalised = bf16( x * rsqrt(mean(x^2) + eps) )   (the sum is accumulated in f32)
//   y          = bf16( weight * normalised )
// One block per row; blockDim.x must be a power of two. Generated PTX lives next to this
// file (rmsnorm.ptx); rebuild it with scripts/build-ptx.sh.
#include <cuda_bf16.h>

extern "C" __global__ void rmsnorm_bf16(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ y,
                                        int hidden, float eps) {
    extern __shared__ float partial[];  // blockDim.x floats
    const size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * hidden;
    __nv_bfloat16* yr = y + row * hidden;

    float sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float normalised = __bfloat162float(__float2bfloat16(__bfloat162float(xr[i]) * inv));
        yr[i] = __float2bfloat16(__bfloat162float(w[i]) * normalised);
    }
}

// The residual add and the RMSNorm that follows it, in one pass: h = bf16(h + delta) in place, then
// y = RMSNorm(h) * w exactly as rmsnorm_bf16 computes it (same reduction order, same roundings), so the result
// is bit-identical to running add_bf16 and rmsnorm_bf16 one after the other. One block per row; blockDim.x
// must be a power of two and equal to what rmsnorm_bf16 is launched with.
extern "C" __global__ void add_rmsnorm_bf16(__nv_bfloat16* __restrict__ h,
                                            const __nv_bfloat16* __restrict__ delta,
                                            const __nv_bfloat16* __restrict__ w,
                                            __nv_bfloat16* __restrict__ y,
                                            int hidden, float eps) {
    extern __shared__ float partial[];  // blockDim.x floats
    const size_t row = blockIdx.x;
    __nv_bfloat16* hr = h + row * hidden;
    const __nv_bfloat16* dr = delta + row * hidden;
    __nv_bfloat16* yr = y + row * hidden;

    float sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const __nv_bfloat16 s = __float2bfloat16(__bfloat162float(hr[i]) + __bfloat162float(dr[i]));
        hr[i] = s;  // the same thread reads this element again below
        const float v = __bfloat162float(s);
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float normalised = __bfloat162float(__float2bfloat16(__bfloat162float(hr[i]) * inv));
        yr[i] = __float2bfloat16(__bfloat162float(w[i]) * normalised);
    }
}
