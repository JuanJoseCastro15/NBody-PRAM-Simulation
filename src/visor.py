import argparse
import itertools
import subprocess
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="total de cuerpos (incluye las 2 masas dominantes)")
ap.add_argument("--pasos", type=int, default=2, help="pasos de simulación por cuadro")
ap.add_argument("--semilla", type=int, default=42)
ap.add_argument("--limite", type=float, default=40.0, help="mitad del ancho de la vista")
ap.add_argument("--cuadros", type=int, default=0, help="número de cuadros (0 = sin límite)")
ap.add_argument("--grabar", default="", help="guardar en un archivo (p. ej. demo.mp4) en vez de mostrar")
args = ap.parse_args()

n = args.n
if n == 0:
    n = int(input("¿Cuántos cuerpos? (recomendado 2000-5000): "))
n = max(3, n)
if args.grabar and not args.cuadros:
    sys.exit("Para grabar indica cuántos cuadros: --cuadros 600")

proc = subprocess.Popen(
    ["./build/stream", str(n), str(args.pasos), str(args.semilla)],
    stdout=subprocess.PIPE,
)


def leer(nbytes):
    buf = proc.stdout.read(nbytes)
    return buf if len(buf) == nbytes else None


hdr = leer(4)
if hdr is None:
    sys.exit("stream no arrancó. ¿Compilaste? Revisa los mensajes de error de arriba.")
N = int(np.frombuffer(hdr, dtype=np.int32)[0])

# Colores: galaxia 0 azul, galaxia 1 naranja (misma división que stream.cu)
n_est = N - 2
n0 = n_est // 2
colores = np.array([[0.3, 0.6, 1.0, 0.8]] * n0 + [[1.0, 0.6, 0.2, 0.8]] * (n_est - n0))

L = args.limite
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

estrellas = ax.scatter(np.zeros(n_est), np.zeros(n_est), s=3, c=colores, linewidths=0)
masas = ax.scatter([0, 0], [0, 0], s=100, c=["red", "yellow"], edgecolors="white", zorder=5)
rastro0, = ax.plot([], [], color="red", lw=1, alpha=0.7)
rastro1, = ax.plot([], [], color="yellow", lw=1, alpha=0.7)
info = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
               va="top", family="monospace", fontsize=10)
artistas = (estrellas, masas, rastro0, rastro1, info)

h0, h1 = deque(maxlen=500), deque(maxlen=500)
t_prev = [time.perf_counter()]


def actualizar(_):
    buf = leer(8 + 8 * N)
    if buf is None:  # la simulación terminó
        plt.close(fig)
        return artistas
    paso = int(np.frombuffer(buf, dtype=np.int32, count=1, offset=0)[0])
    ms = float(np.frombuffer(buf, dtype=np.float32, count=1, offset=4)[0])
    x = np.frombuffer(buf, dtype=np.float32, count=N, offset=8)
    y = np.frombuffer(buf, dtype=np.float32, count=N, offset=8 + 4 * N)

    estrellas.set_offsets(np.column_stack((x[2:], y[2:])))
    masas.set_offsets(np.column_stack((x[:2], y[:2])))
    h0.append((x[0], y[0]))
    h1.append((x[1], y[1]))
    rastro0.set_data(*zip(*h0))
    rastro1.set_data(*zip(*h1))

    ahora = time.perf_counter()
    fps = 1.0 / max(ahora - t_prev[0], 1e-9)
    t_prev[0] = ahora
    info.set_text(f"N = {N}\npaso = {paso}\nGPU = {ms:.3f} ms/paso\nvista = {fps:4.1f} fps")
    return artistas


fig.canvas.mpl_connect("close_event", lambda e: proc.terminate())
frames = range(args.cuadros) if args.cuadros else itertools.count()
ani = FuncAnimation(fig, actualizar, frames=frames, interval=1,
                    blit=True, cache_frame_data=False, repeat=False)

if args.grabar:
    ani.save(args.grabar, fps=30, dpi=120)  # requiere ffmpeg
    proc.terminate()
else:
    plt.show()