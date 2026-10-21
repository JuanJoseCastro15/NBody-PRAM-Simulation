import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("trayectoria.csv")

# Figura 1: trayectoria de las dos masas (vista x-y)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(d.x0, d.y0, label="masa 0")
ax.plot(d.x1, d.y1, label="masa 1")
ax.scatter([d.x0.iloc[0], d.x1.iloc[0]], [d.y0.iloc[0], d.y1.iloc[0]],
           marker="o", color="black", label="inicio")
ax.scatter([d.x0.iloc[-1], d.x1.iloc[-1]], [d.y0.iloc[-1], d.y1.iloc[-1]],
           marker="x", color="red", label="final")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_title("Trayectoria de las dos masas dominantes")
ax.legend()
fig.savefig("trayectoria.png", dpi=150)

# Figura 2: distancia entre las masas según el paso
dist = np.sqrt((d.x0 - d.x1)**2 + (d.y0 - d.y1)**2 + (d.z0 - d.z1)**2)
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(d.paso, dist)
ax2.set_xlabel("paso"); ax2.set_ylabel("distancia entre masas")
ax2.set_title("Separación de las masas dominantes")
fig2.savefig("separacion.png", dpi=150)

print("Guardados: trayectoria.png y separacion.png")