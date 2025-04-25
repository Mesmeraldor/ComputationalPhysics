import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib


def Aligned_spins(N):
    '''
    Returns a (N,N,2) numpy array corresponding to a configuration of aligned spins in a random direction

    Arguments:
        N : (int) number of spins along one direction. N*N being the total number of spins
    '''
    orientation = np.random.uniform(0, 2*np.pi)
    return np.ones((N,N,2))*[np.cos(orientation), np.sin(orientation)]


@njit(parallel=True)
def Run(spins, N, T, steps):
    # calculate physical quantities
    energy = 0
    for i in range(N):
        for j in range(N):
            energy += - np.sum(spins[i][j]*spins[(i+1)%N][j]) - np.sum(spins[i][j]*spins[i][(j+1)%N])
    magnetization = np.sum(np.sum(spins, axis=0), axis=0)

    # create arrays to store physical quantities
    energy_list = np.zeros(steps+1)
    magnetization_list = np.zeros((steps+1, 2))

    # store initial condition
    energy_list[0] = energy
    magnetization_list[0] = magnetization

    for i in range(steps):
        for _ in range(N**2):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)

            orientation = 1. * np.random.uniform(-np.pi, np.pi)
            cosa, sina = spins[i,j,0], spins[i,j,1]
            cosb, sinb = np.cos(orientation), np.sin(orientation)
            new_spin = np.array([cosa*cosb - sina*sinb, sina*cosb + cosa*sinb])
            energy_before = - np.sum(spins[i][j]*spins[(i+1)%N][j]) - np.sum(spins[i][j]*spins[i][(j+1)%N]) - np.sum(spins[i][j]*spins[(i-1)%N][j]) - np.sum(spins[i][j]*spins[i][(j-1)%N])
            energy_after = - np.sum(new_spin*spins[(i+1)%N][j]) - np.sum(new_spin*spins[i][(j+1)%N]) - np.sum(new_spin*spins[(i-1)%N][j]) - np.sum(new_spin*spins[i][(j-1)%N])
            delta_e = energy_after - energy_before

            if delta_e <= 0:
                probability = 1
            else:
                probability = np.exp(-delta_e/T)
            cursor = np.random.uniform(0,1)
            if cursor < probability:
                energy += delta_e
                magnetization += new_spin - spins[i,j]

                spins[i,j] = new_spin

        energy_list[i+1] = energy
        magnetization_list[i+1] = magnetization

    return energy_list, magnetization_list, spins


def vector_to_rgb(angle):
    '''
    Get the rgb value for the given `angle`

    Parameters:
    angle : (float) The angle in radians

    Returns: (array_like) The rgb value as a tuple with values [0...1]
    '''

    # normalize angle
    angle = angle % (2 * np.pi)

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 1, 1))


if __name__ == "__main__":
    N = 50
    T = 1.1

    spins = Aligned_spins(N)

    steps = 10000

    energies, magnetizations, spins = Run(spins, N, T, steps)

    fig, ax = plt.subplots(2)
    fig.suptitle("T = " + str(round(T,1)))
    ax[0].plot(energies / N**2)
    ax[0].title.set_text("Energy per unit")
    ax[0].set_ylim([-4,4])
    ax[1].plot(np.sqrt(np.sum(magnetizations**2,axis=-1)) / N**2)
    ax[1].title.set_text("Magnetization per unit")
    ax[1].set_ylim([-1,1])
    plt.tight_layout()
    plt.show()

    transition = np.reshape(spins, (N*N, 2))
    U = np.reshape(transition[:,0], (N,N))
    V = np.reshape(transition[:,1], (N,N))
    angles = np.arctan2(V, U)

    c3 = np.array(list(map(vector_to_rgb, angles.flatten())))

    fig, ax = plt.subplots()
    q = ax.quiver(U, V, color=c3)
    plt.show()