#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

#include "nbody.cuh"

using namespace nbody;
using Clock = std::chrono::steady_clock;

// Dos masas dominantes que se acercan + estrellas en una esfera (escenario de juguete)
static State escenario_colision(int n_estrellas, unsigned semilla = 42) {
    const int n = n_estrellas + 2;
    State s;
    s.resize(n);
    std::mt19937 rng(semilla);
    std::uniform_real_distribution<float> U(-1.f, 1.f);

    s.x[0] = -15.f; s.y[0] = -2.f; s.vx[0] =  0.5f; s.m[0] = 1000.f;
    s.x[1] =  15.f; s.y[1] =  2.f; s.vx[1] = -0.5f; s.m[1] = 1000.f;

    for (int i = 2; i < n; ++i) {
        float px, py, pz;
        do { px = U(rng); py = U(rng); pz = U(rng); } while (px*px + py*py + pz*pz > 1.f);
        s.x[i] = 20.f * px; s.y[i] = 20.f * py; s.z[i] = 20.f * pz;
        s.vx[i] = 0.1f * U(rng); s.vy[i] = 0.1f * U(rng); s.vz[i] = 0.1f * U(rng);
        s.m[i] = 1.f;
    }
    return s;
}

static float max_diff(const State& a, const State& b, float& escala) {
    float d = 0.f;
    escala = 0.f;
    for (int i = 0; i < a.n; ++i) {
        const float va[6] = {a.x[i], a.y[i], a.z[i], a.vx[i], a.vy[i], a.vz[i]};
        const float vb[6] = {b.x[i], b.y[i], b.z[i], b.vx[i], b.vy[i], b.vz[i]};
        for (int k = 0; k < 6; ++k) {
            d = std::max(d, std::fabs(va[k] - vb[k]));
            escala = std::max(escala, std::fabs(va[k]));
        }
    }
    return d;
}

static void run_ram(const State& init, const Params& p, int pasos, State& res, double& ms) {
    State a = init, b = init;
    const auto t0 = Clock::now();
    for (int s = 0; s < pasos; ++s) { step_ram(a, b, p); std::swap(a, b); }
    ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    res = a;
}

static bool verificar(const Params& p) {
    const int tamanos[] = {2, 5, 127, 128, 129, 1000};
    bool ok = true;
    std::printf("== Verificación GPU vs RAM (5 pasos) ==\n");
    for (int n : tamanos) {
        const State s0 = escenario_colision(n - 2);
        State r_cpu, r_gpu;
        double ms_c; float ms_g;
        run_ram(s0, p, 5, r_cpu, ms_c);
        run_gpu(s0, p, 5, r_gpu, ms_g);
        float escala;
        const float d = max_diff(r_cpu, r_gpu, escala);
        const bool pasa = d <= 1e-3f * escala;
        std::printf("N=%6d  dif.máx=%.3e  (escala %.2f)  %s\n", n, d, escala, pasa ? "OK" : "FALLA");
        ok = ok && pasa;
    }
    return ok;
}

static void benchmark(const Params& p) {
    std::printf("\n== Rendimiento (tiempo por paso) ==\n");
    std::printf("%8s %12s %12s %10s %16s\n", "N", "RAM [ms]", "GPU [ms]", "Speedup", "ns/interaccion");
    const int tamanos[] = {1024, 2048, 4096, 8192, 16384, 32768};
    for (int n : tamanos) {
        const State s0 = escenario_colision(n - 2);
        State r;
        float ms_g;
        run_gpu(s0, p, 50, r, ms_g);                 // calentamiento
        const int pasos = (n <= 4096) ? 200 : 20;
        run_gpu(s0, p, pasos, r, ms_g);
        const double gpu_ms = ms_g / pasos;
        const double ns = gpu_ms * 1e6 / ((double)n * n);
        if (n <= 16384) {                        // la RAM en N grande tarda demasiado
            double ram_ms;
            run_ram(s0, p, 1, r, ram_ms);
            std::printf("%8d %12.2f %12.3f %9.1fx %16.4f\n", n, ram_ms, gpu_ms, ram_ms / gpu_ms, ns);
        } else {
            std::printf("%8d %12s %12.3f %10s %16.4f\n", n, "--", gpu_ms, "--", ns);
        }
    }
}

// Guarda la trayectoria de las dos masas dominantes (cada 20 pasos)
static void trayectoria(const Params& p) {
    State s = escenario_colision(4094);  // N = 4096
    std::FILE* f = std::fopen("trayectoria.csv", "w");
    std::fprintf(f, "paso,x0,y0,z0,x1,y1,z1\n");
    for (int c = 0; c <= 100; ++c) {
        std::fprintf(f, "%d,%f,%f,%f,%f,%f,%f\n", c * 20, s.x[0], s.y[0], s.z[0], s.x[1], s.y[1], s.z[1]);
        if (c == 100) break;
        State r; float ms;
        run_gpu(s, p, 20, r, ms);
        s = r;
    }
    std::fclose(f);
    std::printf("Trayectoria guardada en trayectoria.csv\n");
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaFree(nullptr));  // inicializa el contexto antes de medir
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s | SMs=%d | hilos/SM=%d | máx. residentes=%d | B=%d\n", prop.name,
            prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
            prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor, B);
    std::printf("GPU: %s | tamaño de bloque/tesela B=%d\n\n", prop.name, B);

    const Params p{1.0f, 0.01f, 0.01f};  // G, eps^2, dt
    if (argc > 1 && std::strcmp(argv[1], "tray") == 0) { trayectoria(p); return 0; }

    const bool ok = verificar(p);
    benchmark(p);
    return ok ? 0 : 1;
}