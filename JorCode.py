import numpy as np
import matplotlib.pyplot as plt

N = 50
steps = 100
T = 1.5

spins, energy, magnetization = Animate(Aligned_spins(N), steps, T)

transition = np.reshape(spins, (N*N, 2))
U = np.reshape(transition[:,0], (N,N))
V = np.reshape(transition[:,1], (N,N))

fig, ax = plt.subplots(2)
ax[0].plot(np.array(energy) / N**2)
ax[0].title.set_text("Energy per unit")
ax[0].set_ylim([-2,2])
ax[1].plot(np.array(magnetization) / N**2)
ax[1].title.set_text("Magnetization per unit")
ax[1].set_ylim([-1,1])
plt.tight_layout()
plt.show()