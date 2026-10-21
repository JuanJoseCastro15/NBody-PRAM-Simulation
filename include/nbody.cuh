#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error en %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err_));                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace nbody {

constexpr int B = 128;  // tamaño de bloque = tamaño de tesela

struct Params { float G, eps2, dt; };

// Estado en Structure of Arrays (host)
struct State {
    int n = 0;
    std::vector<float> x, y, z, vx, vy, vz, m;
    void resize(int n_) {
        n = n_;
        for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &m}) v->assign(n, 0.0f);
    }
};

// Algoritmo 1 (RAM): un paso. 'out' debe tener las mismas masas que 'in'.
inline void step_ram(const State& in, State& out, const Params& p) {
    const int n = in.n;
    for (int i = 0; i < n; ++i) {
        float ax = 0.f, ay = 0.f, az = 0.f;
        for (int j = 0; j < n; ++j) {
            const float dx = in.x[j] - in.x[i];
            const float dy = in.y[j] - in.y[i];
            const float dz = in.z[j] - in.z[i];
            const float d2 = dx * dx + dy * dy + dz * dz + p.eps2;
            const float s = p.G * in.m[j] / (d2 * std::sqrt(d2));
            ax += s * dx; ay += s * dy; az += s * dz;
        }
        out.vx[i] = in.vx[i] + ax * p.dt;
        out.vy[i] = in.vy[i] + ay * p.dt;
        out.vz[i] = in.vz[i] + az * p.dt;
        out.x[i] = in.x[i] + out.vx[i] * p.dt;
        out.y[i] = in.y[i] + out.vy[i] * p.dt;
        out.z[i] = in.z[i] + out.vz[i] * p.dt;
    }
}

struct DevBuf { float *x, *y, *z, *vx, *vy, *vz; };

// Algoritmo 2 (kernel): un hilo por cuerpo, teselado, lectura de 'in', escritura en 'out'.
__global__ void kernel_allpairs(DevBuf in, DevBuf out, const float* __restrict__ m,
                                int n, Params p) {
    __shared__ float sx[B], sy[B], sz[B], sm[B];
    const int l = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + l;
    const bool activo = (i < n);

    float xi = 0.f, yi = 0.f, zi = 0.f, vxi = 0.f, vyi = 0.f, vzi = 0.f;
    if (activo) {
        xi = in.x[i];   yi = in.y[i];   zi = in.z[i];
        vxi = in.vx[i]; vyi = in.vy[i]; vzi = in.vz[i];
    }
    float ax = 0.f, ay = 0.f, az = 0.f;

    const int T = (n + B - 1) / B;
    for (int t = 0; t < T; ++t) {
        // Fase 1: TODOS los hilos del bloque cargan (activos o no)
        const int j = t * B + l;
        if (j < n) { sx[l] = in.x[j]; sy[l] = in.y[j]; sz[l] = in.z[j]; sm[l] = m[j]; }
        else       { sx[l] = 0.f;     sy[l] = 0.f;     sz[l] = 0.f;     sm[l] = 0.f; }
        __syncthreads();

        // Fase 2: acumular aceleración
        if (activo) {
            #pragma unroll 8
            for (int k = 0; k < B; ++k) {
                const float dx = sx[k] - xi;
                const float dy = sy[k] - yi;
                const float dz = sz[k] - zi;
                const float d2 = dx * dx + dy * dy + dz * dz + p.eps2;
                const float s = p.G * sm[k] / (d2 * sqrtf(d2));
                ax += s * dx; ay += s * dy; az += s * dz;
            }
        }
        __syncthreads();
    }

    // Fase 3: integración y escritura exclusiva
    if (activo) {
        vxi += ax * p.dt; vyi += ay * p.dt; vzi += az * p.dt;
        out.vx[i] = vxi; out.vy[i] = vyi; out.vz[i] = vzi;
        out.x[i] = xi + vxi * p.dt;
        out.y[i] = yi + vyi * p.dt;
        out.z[i] = zi + vzi * p.dt;
    }
}

inline DevBuf dev_alloc(int n) {
    DevBuf d;
    float** ptrs[] = {&d.x, &d.y, &d.z, &d.vx, &d.vy, &d.vz};
    for (auto pp : ptrs) CUDA_CHECK(cudaMalloc(pp, n * sizeof(float)));
    return d;
}
inline void dev_free(DevBuf& d) {
    float* ptrs[] = {d.x, d.y, d.z, d.vx, d.vy, d.vz};
    for (auto pp : ptrs) CUDA_CHECK(cudaFree(pp));
}
inline void dev_upload(const DevBuf& d, const State& s) {
    const size_t b = s.n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d.x,  s.x.data(),  b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.y,  s.y.data(),  b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z,  s.z.data(),  b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, s.vx.data(), b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, s.vy.data(), b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, s.vz.data(), b, cudaMemcpyHostToDevice));
}
inline void dev_download(const DevBuf& d, State& s) {
    const size_t b = s.n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(s.x.data(),  d.x,  b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.y.data(),  d.y,  b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.z.data(),  d.z,  b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.vx.data(), d.vx, b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.vy.data(), d.vy, b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.vz.data(), d.vz, b, cudaMemcpyDeviceToHost));
}

// Algoritmo 3 (anfitrión): ciclo temporal con doble búfer. 'ms' = tiempo de los pasos.
inline void run_gpu(const State& init, const Params& p, int pasos, State& res, float& ms) {
    const int n = init.n;
    DevBuf a = dev_alloc(n), b = dev_alloc(n);
    float* dm;
    CUDA_CHECK(cudaMalloc(&dm, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dm, init.m.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    dev_upload(a, init);

    const int bloques = (n + B - 1) / B;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int s = 0; s < pasos; ++s) {
        kernel_allpairs<<<bloques, B>>>(a, b, dm, n, p);
        std::swap(a, b);  // el fin de cada kernel es la barrera global
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    res = init;  // copia las masas
    dev_download(a, res);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dm));
    dev_free(a);
    dev_free(b);
}

}  // namespace nbody