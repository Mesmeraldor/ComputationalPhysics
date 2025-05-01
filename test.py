import numpy as np
import matplotlib.pyplot as plt

N=2
orientation = 5

angles = np.random.uniform(-np.pi, np.pi, size=(N,N))
spins = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

print(np.arctan2(-1,0))