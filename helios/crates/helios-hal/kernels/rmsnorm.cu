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
