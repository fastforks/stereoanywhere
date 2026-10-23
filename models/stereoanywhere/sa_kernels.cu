// Stereo Anywhere mono-volume hourglass, full-resolution tail (8-channel NDHWC fp16 volumes, batch 1).
// Size-independent kernels, compiled once (models/stereoanywhere/kernels.py; prebuilt fatbin shipped), launched on
// torch's current stream. Volumes may be D-chunks: a buffer holds global planes [d0, d0 + nd); planes outside [0, D)
// read as zero (= zero padding of the 3x3x3 convs).
//
// mvol_y0   masked volume M[d,h,w,n] = vol[h,w,d] * lm[n,h,w] * rm[n,h,d]  (fp16-rounded like the materialized volume)
//           mode 0: write M; mode 1: y0 = W0 . M + trilinear_up(yb) (1x1 conv on cat([M, up(x)]) split in two), write
//           (optional) and accumulate per-channel sum / sum of squares (instance-norm statistics, double atomics)
// conv3_c8  3x3x3 conv, 8 input channels, 8 (or 2 stored) output channels, tensor cores (mma m16n8k16, fp32 accumulation),
//           input prologue = instance norm (statistics from the producer) + LeakyReLU(0.01) [+ feature attention la * ra],
//           optional output statistics.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float lrelu(float v) { return v >= 0.f ? v : 0.01f * v; }

// ------------------------------------------------------------------------------------------------ masked volume / y0
// grid (ceil(D/32), ceil(W/32), H) for planes [a, b): blockIdx.x tiles d in [a, b). block (32, 8).
extern "C" __global__ void __launch_bounds__(256, 2) mvol_y0(const float* __restrict__ vol,      // [H][W][D] (upstream mono_corr_volume, B=1)
                                   const float* __restrict__ lm,       // [8][H][W]
                                   const float* __restrict__ rm,       // [8][H][D]
                                   int D, int H, int W, int a, int b,
                                   int mode,
                                   const float* __restrict__ w0,       // [8][8] (co, ci) fp32
                                   const __half* __restrict__ yb,      // [Di][Hi][Wi][8] low-res branch (1x1 applied)
                                   int Di, int Hi, int Wi,
                                   __half* __restrict__ out, int out_d0, // out [*][H][W][8], plane 0 = global out_d0
                                   double* __restrict__ st)            // [16] sum, sumsq (or null)
{
    __shared__ float tile[32][33];
    __shared__ float sw[64];
    __shared__ float red[2][8][8];
    const int h = blockIdx.z;
    const int dA = a + blockIdx.x * 32, w0i = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 32 + tx;
    if (tid < 64 && w0) sw[tid] = w0[tid];
    for (int r = ty; r < 32; r += 8) {
        int w = w0i + r, d = dA + tx;
        tile[r][tx] = (w < W && d < b) ? vol[((long long)h * W + w) * D + d] : 0.f;
    }
    __syncthreads();
    float s[8], q[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) s[c] = q[c] = 0.f;
    const int w = w0i + tx;
    float lmv[8];
    if (w < W) {
#pragma unroll
        for (int n = 0; n < 8; ++n) lmv[n] = lm[((long long)n * H + h) * W + w];
    }
    for (int r = ty; r < 32; r += 8) {
        int d = dA + r;
        if (w >= W || d >= b) continue;
        __half hv = __float2half(tile[tx][r]);
        float m[8];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            __half t = __hmul(hv, __float2half(lmv[n]));
            t = __hmul(t, __float2half(rm[((long long)n * H + h) * D + d]));
            m[n] = __half2float(t);
        }
        uint4 raw;
        __half* o = reinterpret_cast<__half*>(&raw);
        if (mode == 0) {
#pragma unroll
            for (int n = 0; n < 8; ++n) o[n] = __float2half(m[n]);
        } else {
            // trilinear (align_corners=True) sample of yb at (d, h, w)
            float fd = D > 1 ? d * (float)(Di - 1) / (float)(D - 1) : 0.f;
            float fh = H > 1 ? h * (float)(Hi - 1) / (float)(H - 1) : 0.f;
            float fw = W > 1 ? w * (float)(Wi - 1) / (float)(W - 1) : 0.f;
            int d0 = min((int)fd, Di - 1), h0 = min((int)fh, Hi - 1), x0 = min((int)fw, Wi - 1);
            int d1 = min(d0 + 1, Di - 1), h1 = min(h0 + 1, Hi - 1), x1 = min(x0 + 1, Wi - 1);
            float td = fd - d0, th = fh - h0, tw = fw - x0;
            float up[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) up[c] = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                int dd = (k & 4) ? d1 : d0, hh = (k & 2) ? h1 : h0, ww = (k & 1) ? x1 : x0;
                float wt = ((k & 4) ? td : 1.f - td) * ((k & 2) ? th : 1.f - th) * ((k & 1) ? tw : 1.f - tw);
                uint4 v = *reinterpret_cast<const uint4*>(yb + (((long long)dd * Hi + hh) * Wi + ww) * 8);
                const __half* hv8 = reinterpret_cast<const __half*>(&v);
#pragma unroll
                for (int c = 0; c < 8; ++c) up[c] += wt * __half2float(hv8[c]);
            }
#pragma unroll
            for (int co = 0; co < 8; ++co) {
                float acc = __half2float(__float2half(up[co]));   // upstream: interpolated fp16 tensor
#pragma unroll
                for (int n = 0; n < 8; ++n) acc += sw[co * 8 + n] * m[n];
                __half y = __float2half(acc);
                o[co] = y;
                float yf = __half2float(y);
                s[co] += yf;
                q[co] += yf * yf;
            }
        }
        if (out) *reinterpret_cast<uint4*>(out + (((long long)(d - out_d0) * H + h) * W + w) * 8) = raw;
    }
    if (mode == 1 && st) {
        // reduce over the block: warp shuffle over tx, then over ty in smem
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            for (int off = 16; off; off >>= 1) {
                s[c] += __shfl_xor_sync(0xffffffffu, s[c], off);
                q[c] += __shfl_xor_sync(0xffffffffu, q[c], off);
            }
        }
        if (tx == 0) {
#pragma unroll
            for (int c = 0; c < 8; ++c) { red[0][ty][c] = s[c]; red[1][ty][c] = q[c]; }
        }
        __syncthreads();
        if (tid < 16) {
            int which = tid >> 3, c = tid & 7;
            float acc = 0.f;
            for (int k = 0; k < 8; ++k) acc += red[which][k][c];
            atomicAdd(st + which * 8 + c, (double)acc);
        }
    }
}

// ------------------------------------------------------------------------------------------------ 3x3x3 conv, C = 8
#define TD 6
#define TH 8
#define TW 32
#define SD (TD + 2)
#define SH (TH + 2)
#define SW (TW + 2)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// grid (ceil(W/TW), ceil(H/TH), ceil((ob-oa)/TD)), block 256. Output planes [oa, ob) (global).
extern "C" __global__ void __launch_bounds__(256) conv3_c8(
    const __half* __restrict__ x, int x_d0, int x_nd,      // input buffer [x_nd][H][W][8], plane 0 = global x_d0
    int D, int H, int W, int oa, int ob,
    const double* __restrict__ st_in, double inv_n, float eps,   // input instance-norm statistics (sum, sumsq)
    const __half* __restrict__ la,                          // [H][W][8] or null (feature attention, left)
    const __half* __restrict__ ra,                          // [D][H][8] or null (feature attention, right)
    const uint32_t* __restrict__ wp,                        // [14][2][32] packed half2 B fragments
    __half* __restrict__ y, int y_d0, int co_store,         // output [*][H][W][co_store], plane 0 = global y_d0 (null: no store)
    double* __restrict__ st_out)                            // [16] or null
{
    __shared__ __align__(16) __half sm[SD * SH * SW * 8];
    __shared__ float sc[8], sh_[8];
    __shared__ float red[2][8][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int w0 = blockIdx.x * TW, h0 = blockIdx.y * TH, d0 = oa + blockIdx.z * TD;
    if (tid < 8) {
        double mean = st_in[tid] * inv_n;
        double var = st_in[8 + tid] * inv_n - mean * mean;
        float rstd = rsqrtf((float)fmax(var, 0.0) + eps);
        sc[tid] = rstd;
        sh_[tid] = (float)(-mean) * rstd;
    }
    uint32_t bw[14][2];
#pragma unroll
    for (int j = 0; j < 14; ++j) {
        bw[j][0] = wp[(j * 2 + 0) * 32 + lane];
        bw[j][1] = wp[(j * 2 + 1) * 32 + lane];
    }
    __syncthreads();
    // halo tile load with prologue
    for (int i = tid; i < SD * SH * SW; i += 256) {
        int lw = i % SW, r = i / SW, lh = r % SH, ld = r / SH;
        int gd = d0 + ld - 1, gh = h0 + lh - 1, gw = w0 + lw - 1;
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (gd >= 0 && gd < D && gh >= 0 && gh < H && gw >= 0 && gw < W) {
            raw = *reinterpret_cast<const uint4*>(x + (((long long)(gd - x_d0) * H + gh) * W + gw) * 8);
            __half* hv = reinterpret_cast<__half*>(&raw);
            float av[8], bv[8];
            if (la) {
                uint4 lr = *reinterpret_cast<const uint4*>(la + ((long long)gh * W + gw) * 8);
                uint4 rr = *reinterpret_cast<const uint4*>(ra + ((long long)gd * H + gh) * 8);
                const __half* l8 = reinterpret_cast<const __half*>(&lr);
                const __half* r8 = reinterpret_cast<const __half*>(&rr);
#pragma unroll
                for (int c = 0; c < 8; ++c) { av[c] = __half2float(l8[c]); bv[c] = __half2float(r8[c]); }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float v = lrelu(__half2float(hv[c]) * sc[c] + sh_[c]);
                if (la) v = __half2float(__float2half(v)) * av[c] * bv[c];
                hv[c] = __float2half(v);
            }
        }
        *reinterpret_cast<uint4*>(sm + i * 8) = raw;
    }
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    float s0 = 0.f, s1 = 0.f, q0 = 0.f, q1 = 0.f;
    // TD * TH * 2 m-tiles of 16 voxels (2 per W row of 32), spread over the 8 warps
#pragma unroll 1
    for (int mt = warp; mt < TD * TH * 2; mt += 8) {
        const int ld = mt / (TH * 2), rem = mt % (TH * 2), lh = rem >> 1, lw0 = (rem & 1) * 16;
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 14; ++j) {
            uint32_t af[4];
#pragma unroll
            for (int hk = 0; hk < 2; ++hk) {
                const int tap = 2 * j + hk;
                if (tap < 27) {
                    const int kd = tap / 9, kh = (tap / 3) % 3, kw = tap % 3;
                    const int base = ((ld + kd) * SH + (lh + kh)) * SW + lw0 + kw;
                    af[hk * 2 + 0] = *reinterpret_cast<const uint32_t*>(sm + (base + g) * 8 + 2 * t);
                    af[hk * 2 + 1] = *reinterpret_cast<const uint32_t*>(sm + (base + g + 8) * 8 + 2 * t);
                } else {
                    af[hk * 2 + 0] = 0u;
                    af[hk * 2 + 1] = 0u;
                }
            }
            // fragment order: a0 = (row g, k 2t..), a1 = (row g+8, k 2t..), a2 = (row g, k 2t+8..), a3 = (row g+8, k 2t+8..)
            uint32_t a[4] = {af[0], af[1], af[2], af[3]};
            mma16816(c, a, bw[j][0], bw[j][1]);
        }
        const int gd = d0 + ld, gh = h0 + lh;
        if (gd < ob && gh < H) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int gw = w0 + lw0 + g + 8 * hr;
                if (gw >= W) continue;
                __half2 v = __floats2half2_rn(c[2 * hr], c[2 * hr + 1]);
                if (y && 2 * t < co_store)
                    *reinterpret_cast<__half2*>(y + (((long long)(gd - y_d0) * H + gh) * W + gw) * co_store + 2 * t) = v;
                float2 f = __half22float2(v);
                s0 += f.x; s1 += f.y; q0 += f.x * f.x; q1 += f.y * f.y;
            }
        }
    }
    if (st_out) {
        // reduce over g (lanes with equal t), then over warps
        for (int off = 4; off < 32; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
            q0 += __shfl_xor_sync(0xffffffffu, q0, off);
            q1 += __shfl_xor_sync(0xffffffffu, q1, off);
        }
        if (g == 0) {
            red[0][warp][2 * t] = s0; red[0][warp][2 * t + 1] = s1;
            red[1][warp][2 * t] = q0; red[1][warp][2 * t + 1] = q1;
        }
        __syncthreads();
        if (tid < 16) {
            int which = tid >> 3, ch = tid & 7;
            float acc = 0.f;
            for (int k = 0; k < 8; ++k) acc += red[which][k][ch];
            atomicAdd(st_out + which * 8 + ch, (double)acc);
        }
    }
}

// ------------------------------------------------------------------------------------------------ ConvGRU gates (NHWC)
// Row-major [P][*] fp16 views with row strides (elements); Hd % 8 == 0, strides / offsets multiples of 8.
__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

// zr = conv([h | x]) without bias. z -> zr[:, :Hd] (in place), r * h -> hx[:, :Hd]
extern "C" __global__ void gru_zr(__half* zr, int zs, const float* __restrict__ bzr,
                                  const __half* __restrict__ cz, const __half* __restrict__ cr, int cs,
                                  const __half* __restrict__ h, int hs, __half* hx, int hxs, long long P, int Hd) {
    const int CV = Hd / 8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < P * CV; i += (long long)gridDim.x * blockDim.x) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 8;
        uint4 zv = *reinterpret_cast<const uint4*>(zr + p * zs + c0);
        uint4 rv = *reinterpret_cast<const uint4*>(zr + p * zs + Hd + c0);
        uint4 czv = *reinterpret_cast<const uint4*>(cz + p * cs + c0);
        uint4 crv = *reinterpret_cast<const uint4*>(cr + p * cs + c0);
        uint4 hv = *reinterpret_cast<const uint4*>(h + p * hs + c0);
        __half* z8 = reinterpret_cast<__half*>(&zv);
        const __half* r8 = reinterpret_cast<const __half*>(&rv);
        const __half* cz8 = reinterpret_cast<const __half*>(&czv);
        const __half* cr8 = reinterpret_cast<const __half*>(&crv);
        __half* h8 = reinterpret_cast<__half*>(&hv);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            // upstream rounding: conv(+bias) -> fp16, + c -> fp16, sigmoid -> fp16, r * h -> fp16
            float zz = __half2float(__float2half(__half2float(z8[k]) + bzr[c0 + k]));
            float rr = __half2float(__float2half(__half2float(r8[k]) + bzr[Hd + c0 + k]));
            z8[k] = __float2half(sigm(__half2float(__float2half(zz + __half2float(cz8[k])))));
            float r = __half2float(__float2half(sigm(__half2float(__float2half(rr + __half2float(cr8[k]))))));
            h8[k] = __float2half(r * __half2float(h8[k]));
        }
        *reinterpret_cast<uint4*>(zr + p * zs + c0) = zv;
        *reinterpret_cast<uint4*>(hx + p * hxs + c0) = hv;
    }
}

// q = conv([r*h | x]) without bias. hnew = (1 - z) * h + z * tanh(q + bq + cq) -> out (may alias h)
extern "C" __global__ void gru_out(const __half* __restrict__ q, int qs, const float* __restrict__ bq,
                                   const __half* __restrict__ cq, int cs, const __half* __restrict__ z, int zs,
                                   const __half* h, int hs, __half* out, int os, long long P, int Hd) {
    const int CV = Hd / 8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < P * CV; i += (long long)gridDim.x * blockDim.x) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 8;
        uint4 qv = *reinterpret_cast<const uint4*>(q + p * qs + c0);
        uint4 cv = *reinterpret_cast<const uint4*>(cq + p * cs + c0);
        uint4 zv = *reinterpret_cast<const uint4*>(z + p * zs + c0);
        uint4 hv = *reinterpret_cast<const uint4*>(h + p * hs + c0);
        const __half* q8 = reinterpret_cast<const __half*>(&qv);
        const __half* c8 = reinterpret_cast<const __half*>(&cv);
        const __half* z8 = reinterpret_cast<const __half*>(&zv);
        __half* h8 = reinterpret_cast<__half*>(&hv);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float qq = __half2float(__float2half(__half2float(q8[k]) + bq[c0 + k]));
            float t = __half2float(__float2half(tanhf(__half2float(__float2half(qq + __half2float(c8[k]))))));
            float zf = __half2float(z8[k]), hf = __half2float(h8[k]);
            h8[k] = __float2half((1.f - zf) * hf + zf * t);
        }
        *reinterpret_cast<uint4*>(out + p * os + c0) = hv;
    }
}

// y[p, c] = act(y[p, c] + b[c]) in place (row stride ys; C % 8 == 0 vector path, else scalar). act 0 none, 1 relu
extern "C" __global__ void bias_act(__half* y, int ys, const float* __restrict__ b, long long P, int C, int act) {
    if ((C & 7) == 0 && (ys & 7) == 0) {
        const int CV = C / 8;
        for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < P * CV; i += (long long)gridDim.x * blockDim.x) {
            long long p = i / CV;
            int c0 = (int)(i - p * CV) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(y + p * ys + c0);
            __half* h8 = reinterpret_cast<__half*>(&v);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float f = __half2float(__float2half(__half2float(h8[k]) + b[c0 + k]));
                h8[k] = __float2half(act ? fmaxf(f, 0.f) : f);
            }
            *reinterpret_cast<uint4*>(y + p * ys + c0) = v;
        }
    } else {
        for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < P * C; i += (long long)gridDim.x * blockDim.x) {
            long long p = i / C;
            int c = (int)(i - p * C);
            float f = __half2float(__float2half(__half2float(y[p * ys + c]) + b[c]));
            y[p * ys + c] = __float2half(act ? fmaxf(f, 0.f) : f);
        }
    }
}

// ------------------------------------------------------------------------------------------------ 1D corr lookup
// == CorrBlock1D.__call__ (grid_sample bilinear, align_corners=True, zero padding) for 4 pyramid levels, radius r:
// out[b, h, w, l * (2r+1) + k] = lerp of level l row (b, h, w) at x / 2^l + (k - r), x = coords[b, 0, h, w]. fp16 out.
extern "C" __global__ void corr_lookup(const float* __restrict__ c0, const float* __restrict__ c1,
                                       const float* __restrict__ c2, const float* __restrict__ c3,
                                       int n0, int n1, int n2, int n3,
                                       const float* __restrict__ x, long long sb, long long sh, long long sw,
                                       int B, int H, int W, __half* __restrict__ out, int r) {
    const long long P = (long long)B * H * W;
    const int K = 2 * r + 1;
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P; p += (long long)gridDim.x * blockDim.x) {
        int w = (int)(p % W);
        long long t = p / W;
        int h = (int)(t % H), b = (int)(t / H);
        float xv = x[b * sb + h * sh + w * sw];
        __half* o = out + p * 4 * K;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float* c = l == 0 ? c0 : l == 1 ? c1 : l == 2 ? c2 : c3;
            const int n = l == 0 ? n0 : l == 1 ? n1 : l == 2 ? n2 : n3;
            const float* row = c + p * n;
            float xs = xv / (float)(1 << l);
            for (int k = 0; k < K; ++k) {
                float s = xs + (float)(k - r);
                float f = floorf(s);
                float a = s - f;
                int i0 = (int)f, i1 = i0 + 1;
                float v0 = (i0 >= 0 && i0 < n) ? row[i0] : 0.f;
                float v1 = (i1 >= 0 && i1 < n) ? row[i1] : 0.f;
                o[l * K + k] = __float2half(v0 * (1.f - a) + v1 * a);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ norm / act epilogues
// y[n, p, c] = act2( act1(y * s[n, c] + t[n, c]) + (r ? r[n, p, c] * rs[n, c] + rt[n, c] : 0) ), NHWC fp16, in place.
// s / t (and rs / rt) fp32 [N][C] (per-instance statistics or folded batch norm). rs null: residual added as is.
// Intermediate fp16 roundings as the upstream op sequence (norm -> fp16, act, residual add -> fp16).
extern "C" __global__ void norm_act(__half* y, const float* __restrict__ s, const float* __restrict__ t,
                                    const __half* __restrict__ r, const float* __restrict__ rs, const float* __restrict__ rt,
                                    int N, long long P, int C, int act1, int act2) {
    const int CV = C / 8;
    const long long total = (long long)N * P * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long row = i / CV;
        int c0 = (int)(i - row * CV) * 8;
        int n = (int)(row / P);
        uint4 v = *reinterpret_cast<const uint4*>(y + row * C + c0);
        __half* h8 = reinterpret_cast<__half*>(&v);
        uint4 rv;
        const __half* r8 = reinterpret_cast<const __half*>(&rv);
        if (r) rv = *reinterpret_cast<const uint4*>(r + row * C + c0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int c = n * C + c0 + k;
            float f = __half2float(__float2half(__half2float(h8[k]) * s[c] + t[c]));
            if (act1) f = fmaxf(f, 0.f);
            if (r) {
                float rr = __half2float(r8[k]);
                if (rs) rr = __half2float(__float2half(rr * rs[c] + rt[c]));
                f = __half2float(__float2half(f + rr));
                if (act2) f = fmaxf(f, 0.f);
            }
            h8[k] = __float2half(f);
        }
        *reinterpret_cast<uint4*>(y + row * C + c0) = v;
    }
}

// ------------------------------------------------------------------------------------------------ disparity / confidence
// From the 2-channel classifier output o[d][h][w][2] (d = W3, w = W2; ch 0 disparity logits, ch 1 confidence logits):
// left (softmax over d for each (h, w)):  disp = w - E[d], conf = 1 - entropy / log2(D)
// right (softmax over w for each (d, h)): disp = E[w] - d, conf = 1 - entropy / log2(W)
// entropy = -sum p log2(p + 1e-6) (upstream estimate_*_confidence), fp32, two passes (max / sum, then moments).
extern "C" __global__ void est_left(const __half* __restrict__ o, int D, int H, int W, float* __restrict__ disp,
                                    float* __restrict__ conf) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)H * W) return;
    const long long st = (long long)H * W;          // d stride in half2 units
    const __half2* p = reinterpret_cast<const __half2*>(o) + i;
    float m0 = -INFINITY, m1 = -INFINITY;
    for (int d = 0; d < D; ++d) {
        float2 v = __half22float2(p[d * st]);
        m0 = fmaxf(m0, v.x); m1 = fmaxf(m1, v.y);
    }
    float s0 = 0.f, e0 = 0.f, s1 = 0.f;
    for (int d = 0; d < D; ++d) {
        float2 v = __half22float2(p[d * st]);
        float a = __expf(v.x - m0);
        s0 += a; e0 += a * (float)d;
        s1 += __expf(v.y - m1);
    }
    float inv1 = 1.f / s1, ent = 0.f;
    for (int d = 0; d < D; ++d) {
        float q = __expf(__half2float(p[d * st].y) - m1) * inv1;
        ent += q * log2f(q + 1e-6f);
    }
    int w = (int)(i % W);
    disp[i] = (float)w - e0 / s0;
    conf[i] = 1.f + ent / log2f((float)D);
}

// one warp per (d, h) row; out index d * H + h... stored as [h][d] (upstream layout B 1 H W3)
extern "C" __global__ void est_right(const __half* __restrict__ o, int D, int H, int W, float* __restrict__ disp,
                                     float* __restrict__ conf) {
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= (long long)D * H) return;
    const __half2* p = reinterpret_cast<const __half2*>(o) + row * W;
    float m0 = -INFINITY, m1 = -INFINITY;
    for (int w = lane; w < W; w += 32) {
        float2 v = __half22float2(p[w]);
        m0 = fmaxf(m0, v.x); m1 = fmaxf(m1, v.y);
    }
    for (int off = 16; off; off >>= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
    }
    float s0 = 0.f, e0 = 0.f, s1 = 0.f;
    for (int w = lane; w < W; w += 32) {
        float2 v = __half22float2(p[w]);
        float a = __expf(v.x - m0);
        s0 += a; e0 += a * (float)w;
        s1 += __expf(v.y - m1);
    }
    for (int off = 16; off; off >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        e0 += __shfl_xor_sync(0xffffffffu, e0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
    }
    float inv1 = 1.f / s1, ent = 0.f;
    for (int w = lane; w < W; w += 32) {
        float q = __expf(__half2float(p[w].y) - m1) * inv1;
        ent += q * log2f(q + 1e-6f);
    }
    for (int off = 16; off; off >>= 1) ent += __shfl_xor_sync(0xffffffffu, ent, off);
    if (lane == 0) {
        int d = (int)(row / H), h = (int)(row % H);
        disp[(long long)h * D + d] = e0 / s0 - (float)d;
        conf[(long long)h * D + d] = 1.f + ent / log2f((float)W);
    }
}

// ------------------------------------------------------------------------------------------------ 3D norm + act + attention
// y[d, h, w, c] = lrelu(y * s[c] + t[c]) [* la[h, w, c] * ra[d, h, c]]  in place, NDHWC fp16 (batch 1), C % 8 == 0
extern "C" __global__ void norm_lrelu_att(__half* y, const float* __restrict__ s, const float* __restrict__ t,
                                          const __half* __restrict__ la, const __half* __restrict__ ra,
                                          int D, int H, int W, int C) {
    const int CV = C / 8;
    const long long total = (long long)D * H * W * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long vox = i / CV;
        int c0 = (int)(i - vox * CV) * 8;
        int w = (int)(vox % W);
        long long r = vox / W;
        int h = (int)(r % H), d = (int)(r / H);
        uint4 v = *reinterpret_cast<const uint4*>(y + vox * C + c0);
        __half* h8 = reinterpret_cast<__half*>(&v);
        uint4 lv, rv;
        if (la) {
            lv = *reinterpret_cast<const uint4*>(la + ((long long)h * W + w) * C + c0);
            rv = *reinterpret_cast<const uint4*>(ra + ((long long)d * H + h) * C + c0);
        }
        const __half* l8 = reinterpret_cast<const __half*>(&lv);
        const __half* r8 = reinterpret_cast<const __half*>(&rv);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float f = lrelu(__half2float(h8[k]) * s[c0 + k] + t[c0 + k]);
            if (la) f = __half2float(__float2half(f)) * __half2float(l8[k]) * __half2float(r8[k]);
            h8[k] = __float2half(f);
        }
        *reinterpret_cast<uint4*>(y + vox * C + c0) = v;
    }
}

// ------------------------------------------------------------------------------------------------ mirror truncation
// vol[h, w, d] *= (1 - c) + c * (sigmoid((w - disp[h, w]) - d) * (1 - att) + att), c = conf[h, w]  (truncate_corr_volume_v2
// with conf_th None, applied to the stereo correlation volume in place; fp32, same op order)
extern "C" __global__ void trunc_mul(float* vol, const float* __restrict__ disp, const float* __restrict__ conf,
                                     int H, int W, int D, float att) {
    const long long total = (long long)H * W * D;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        int d = (int)(i % D);
        long long hw = i / D;
        int w = (int)(hw % W);
        float c = conf[hw];
        float x = ((float)w - disp[hw]) - (float)d;
        float sg = 1.f / (1.f + expf(-x));
        float m = 1.f * (1.f - c) + c * (sg * (1.f - att) + att);
        vol[i] = m * vol[i];
    }
}

// ------------------------------------------------------------------------------------------------ GRU state resampling
// NHWC fp16 rows (batch 1): src [Hs][Ws][C] (row stride ss), dst [Hd][Wd][C] (row stride ds), C % 8 == 0
// bilinear, align_corners=True (= F.interpolate on the fp16 tensor: fp32 math, one rounding)
extern "C" __global__ void bilerp_rows(const __half* __restrict__ src, int ss, int Hs, int Ws, __half* dst, int ds,
                                       int Hd, int Wd, int C) {
    const int CV = C / 8;
    const long long total = (long long)Hd * Wd * CV;
    const float sh = Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.f;
    const float sw = Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd), y = (int)(pix / Wd);
        float fy = sh * y, fx = sw * x;
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        uint4 a = *reinterpret_cast<const uint4*>(src + ((long long)y0 * Ws + x0) * ss + c0);
        uint4 b = *reinterpret_cast<const uint4*>(src + ((long long)y0 * Ws + x1) * ss + c0);
        uint4 c = *reinterpret_cast<const uint4*>(src + ((long long)y1 * Ws + x0) * ss + c0);
        uint4 d = *reinterpret_cast<const uint4*>(src + ((long long)y1 * Ws + x1) * ss + c0);
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * ds + c0) = o;
    }
}

// avg_pool2d(3, stride 2, padding 1), count_include_pad (divide by 9)
extern "C" __global__ void pool2x_rows(const __half* __restrict__ src, int ss, int Hs, int Ws, __half* dst, int ds,
                                       int Hd, int Wd, int C) {
    const int CV = C / 8;
    const long long total = (long long)Hd * Wd * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd), y = (int)(pix / Wd);
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = 0.f;
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = 2 * y + dy;
            if (yy < 0 || yy >= Hs) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = 2 * x + dx;
                if (xx < 0 || xx >= Ws) continue;
                uint4 v = *reinterpret_cast<const uint4*>(src + ((long long)yy * Ws + xx) * ss + c0);
                const __half* v8 = reinterpret_cast<const __half*>(&v);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __half2float(v8[k]);
            }
        }
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k) o8[k] = __float2half(acc[k] / 9.f);
        *reinterpret_cast<uint4*>(dst + pix * ds + c0) = o;
    }
}

// bilinear resize NHWC fp16 (contiguous), any batch: src index = scale * dst (align_corners) or
// max(scale * (dst + 0.5) - 0.5, 0); scales computed by the caller as PyTorch does (fp32). Same arithmetic as
// upsample_bilinear2d_nhwc_out_frame.
extern "C" __global__ void bilerp_nhwc(const __half* __restrict__ src, int N, int Hs, int Ws, __half* __restrict__ dst,
                                       int Hd, int Wd, int C, float sh, float sw, int align) {
    const int CV = C / 8;
    const long long total = (long long)N * Hd * Wd * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd);
        long long r = pix / Wd;
        int y = (int)(r % Hd), n = (int)(r / Hd);
        float fy = align ? sh * y : fmaxf(sh * (y + 0.5f) - 0.5f, 0.f);
        float fx = align ? sw * x : fmaxf(sw * (x + 0.5f) - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* s = src + (long long)n * Hs * Ws * C + c0;
        uint4 a = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x0) * C);
        uint4 b = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x1) * C);
        uint4 c = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x0) * C);
        uint4 d = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x1) * C);
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = o;
    }
}

// ------------------------------------------------------------------------------------------------ context-encoder stem
// y[h, w, :] = relu((conv7x7(x)[h, w, :] + b) * s + t), 1 input channel fp32 (the mono depth), 64 outputs fp16 NHWC,
// zero padding 3, stride 1. ws [64][49] fp32 (input-channel-summed kernel); conv + bias rounded to fp16, then BN affine.
#define ST_T 16
extern "C" __global__ void __launch_bounds__(256) stem7_c64(const float* __restrict__ x, int H, int W,
                                                            const float* __restrict__ ws, const float* __restrict__ bias,
                                                            const float* __restrict__ sc, const float* __restrict__ sh,
                                                            __half* __restrict__ y) {
    __shared__ float tile[ST_T + 6][ST_T + 6];
    __shared__ float wsm[64 * 49];
    const int tid = threadIdx.y * ST_T + threadIdx.x;
    const int h0 = blockIdx.y * ST_T, w0 = blockIdx.x * ST_T;
    for (int i = tid; i < 64 * 49; i += 256) wsm[i] = ws[i];
    for (int i = tid; i < (ST_T + 6) * (ST_T + 6); i += 256) {
        int r = i / (ST_T + 6), c = i % (ST_T + 6);
        int gh = h0 + r - 3, gw = w0 + c - 3;
        tile[r][c] = (gh >= 0 && gh < H && gw >= 0 && gw < W) ? x[(long long)gh * W + gw] : 0.f;
    }
    __syncthreads();
    const int h = h0 + threadIdx.y, w = w0 + threadIdx.x;
    if (h >= H || w >= W) return;
    float v[49];
#pragma unroll
    for (int k = 0; k < 49; ++k) v[k] = tile[threadIdx.y + k / 7][threadIdx.x + k % 7];
    __half* out = y + ((long long)h * W + w) * 64;
#pragma unroll 1
    for (int g = 0; g < 8; ++g) {
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float a = 0.f;
            const float* wr = wsm + (g * 8 + j) * 49;
#pragma unroll
            for (int k = 0; k < 49; ++k) a = fmaf(wr[k], v[k], a);
            acc[j] = a;
        }
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = g * 8 + j;
            // upstream: fp32 conv (+bias) -> fp16 -> BN affine -> relu (cmix rounds the conv output to fp16 first)
            float cv = __half2float(__float2half(acc[j] + bias[c]));
            o8[j] = __float2half(fmaxf(cv * sc[c] + sh[c], 0.f));
        }
        *reinterpret_cast<uint4*>(out + g * 8) = o;
    }
}
