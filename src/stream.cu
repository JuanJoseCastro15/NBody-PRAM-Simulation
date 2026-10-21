// Simulación en vivo: escribe cada cuadro (posiciones x, y) en stdout, en binario.
// IMPORTANTE: no imprimir nada más en stdout; los mensajes van a stderr.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "nbody.cuh"

using namespace nbody;

// Dos "galaxias": cada masa dominante tiene ~n/2 estrellas en órbita circular.
static State escenario_galaxias(int n_total, unsigned semilla, const Params& p) {
    State s;
    s.resize(n_total);

    // Masas dominantes: posiciones y velocidades predeterminadas
    s.x[0] = -20.f; s.y[0] = -5.f; s.vx[0] =  1.0f; s.m[0] = 1000.f;
    s.x[1] =  20.f; s.y[1] =  5.f; s.vx[1] = -1.0f; s.m[1] = 1000.f;

    const int n_est = n_total - 2;
    const int n0 = n_est / 2;  // estrellas de la galaxia 0 (el visor usa la misma división)
    std::mt19937 rng(semilla);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float DOS_PI = 6.2831853f;

    for (int k = 0; k < n_est; ++k) {
        const int i = k + 2;
        const int g = (k < n0) ? 0 : 1;  // a qué masa orbita
        const float r  = 3.f + 7.f * std::sqrt(U(rng));
        const float th = DOS_PI * U(rng);
        const float dz = 0.6f * (U(rng) - 0.5f);
        // Velocidad circular alrededor de la masa dominante (con suavizado)
        const float v = std::sqrt(p.G * s.m[g] * r * r / std::pow(r * r + p.eps2, 1.5f));

        s.x[i]  = s.x[g] + r * std::cos(th);
        s.y[i]  = s.y[g] + r * std::sin(th);
        s.z[i]  = s.z[g] + dz;
        s.vx[i] = s.vx[g] - v * std::sin(th);
        s.vy[i] = s.vy[g] + v * std::cos(th);
        s.vz[i] = 0.f;
        s.m[i]  = 1.f;
    }
    return s;
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 3000;
    const int k = (argc > 2) ? std::atoi(argv[2]) : 2;  // pasos por cuadro
    const unsigned semilla = (argc > 3) ? (unsigned)std::atoi(argv[3]) : 42u;
    if (n < 3 || k < 1) {
        std::fprintf(stderr, "uso: stream N(>=3) [pasos_por_cuadro] [semilla]\n");
        return 1;
    }

    const Params p{1.0f, 0.01f, 0.01f};  // G, eps^2, dt (mismos que demo.cu)
    const State s0 = escenario_galaxias(n, semilla, p);

    DevBuf a = dev_alloc(n), b = dev_alloc(n);
    float* dm;
    CUDA_CHECK(cudaMalloc(&dm, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dm, s0.m.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    dev_upload(a, s0);

    const int bloques = (n + B - 1) / B;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<float> hx(n), hy(n);

    // Protocolo: [int32 N] y luego, por cada cuadro:
    // [int32 paso][float32 ms_por_paso][N float32 x][N float32 y]
    const int32_t n32 = n;
    std::fwrite(&n32, sizeof(n32), 1, stdout);
    std::fflush(stdout);
    std::fprintf(stderr, "stream: N=%d, %d pasos/cuadro, semilla=%u\n", n, k, semilla);

    auto escribir = [](const void* ptr, size_t tam, size_t cant) {
        return std::fwrite(ptr, tam, cant, stdout) == cant;
    };

    int32_t paso = 0;
    while (true) {
        CUDA_CHECK(cudaEventRecord(e0));
        for (int s = 0; s < k; ++s) {
            kernel_allpairs<<<bloques, B>>>(a, b, dm, n, p);
            std::swap(a, b);  // doble búfer
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        paso += k;

        CUDA_CHECK(cudaMemcpy(hx.data(), a.x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), a.y, n * sizeof(float), cudaMemcpyDeviceToHost));

        const float ms_paso = ms / k;
        if (!(escribir(&paso, sizeof(paso), 1) && escribir(&ms_paso, sizeof(ms_paso), 1) &&
              escribir(hx.data(), sizeof(float), n) && escribir(hy.data(), sizeof(float), n)))
            break;
        if (std::fflush(stdout) != 0) break;  // el visor cerró la tubería
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dm));
    dev_free(a);
    dev_free(b);
    return 0;
}