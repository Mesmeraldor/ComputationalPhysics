import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def Aligned_spins(N):
    '''
    Returns a (N,N,2) numpy array corresponding to a configuration of aligned spins in a random direction

    Arguments:
        N : (int) number of spins along one direction. N*N being the total number of spins
    '''
    orientation = np.random.uniform(0, 2*np.pi)
    return np.ones((N,N,2))*[np.cos(orientation), np.sin(orientation)]


def Random_spins(N):
    '''
    Returns a (N,N,2) numpy array corresponding to a configuration of random oriented spins

    Arguments:
        N : (int) number of spins along one direction. N*N being the total number of spins
    '''
    spins = np.zeros((N,N,2))
    for i in range(N):
        for j in range(N):
            orientation = np.random.uniform(0, 2*np.pi)
            spins[i][j] += [np.cos(orientation), np.sin(orientation)]
    return spins


def Configuration(N, T):
    '''
    Returns a (N,N,2) numpy array corresponding to a configuration at equilibrium at temerature T

    Arguments:
        N : (int) number of spins along one direction. N*N being the total number of spins
        T : (float) temperature of the system at equilibrium
    '''
    old_spins = Aligned_spins(N)
    new_spins, delta_e = Update(old_spins, T)
    old_energy = Energy(old_spins)
    new_energy = Energy(new_spins)
    old_magnetization = Magnetization(old_spins)
    new_magnetization = Magnetization(new_spins)
    while abs(new_energy-old_energy) > 0.01 and abs(new_magnetization-old_magnetization) > 0.01:
        old_spins = np.copy(new_spins)
        new_spins, delta_e = Update(new_spins, T)
        old_energy = Energy(old_spins)
        new_energy = Energy(new_spins)
        old_magnetization = Magnetization(old_spins)
        new_magnetization = Magnetization(new_spins)
    return new_spins


def Magnetization(spins):
    '''
    Returns the total magnetization of one configuration of spins as a float

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
    '''
    total_spin = np.sum(spins, (0,1))
    magnetization = np.sqrt(total_spin[0]*total_spin[0] + total_spin[1]*total_spin[1])
    return magnetization


def Energy(spins):
    '''
    Returns the total energy of one configuration of spins as a float

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
    '''
    energy = -np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)))
    return energy


def Update(spins, T):
    '''
    Returns the new configuration of spins after one timestep as a (N,N,2) numpy array,
    and the gain or loss of energy of this change as a float

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
        T: (float) temperature of the system
    '''
    N = len(spins)

    for _ in range(N**2):
        angles = np.random.uniform(-np.pi, np.pi, size=(N,N))
        new_spins = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        new_energies = -np.sum(new_spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)
                                 + np.roll(spins, -1, axis=0) + np.roll(spins, 1, axis=1)), axis=-1)
        old_energies = -np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)
                                 + np.roll(spins, -1, axis=0) + np.roll(spins, 1, axis=1)), axis=-1)
        delta_energies = new_energies - old_energies
        probabilities = np.exp(-delta_energies/T)
        check = np.random.uniform(0, 1, size=(N,N))
        transition = (probabilities > check).astype(int)
        spins = (1 - transition[:, :, np.newaxis]) * spins + transition[:, :, np.newaxis] * new_spins
    return spins, np.sum(delta_energies * transition)


def Animate(spins, steps, T):
    '''
    Returns the new configuration of spins after 'steps' timesteps at temperature T  as a (N,N,2) numpy array
    and the energy and magnetization of this configuration at each time as (steps+1) numpy array

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
        steps: (int) number of timesteps to animate if steps=0 the animation stops at equilibrium
        T: (float) temperature of the system
    '''
    energy = [Energy(spins)]
    magnetization = [Magnetization(spins)]
    for i in range(int(600 - 300 * abs(steps - 1.1))):        #bring the system to equilibrium
        spins, delta_e = Update(spins, T)
    if steps > 0:
        for i in range(steps):
            spins, delta_e = Update(spins, T)
            energy += [energy[i]+delta_e]
            magnetization += [Magnetization(spins)]
    return spins, np.array(energy), np.array(magnetization)


def CorrelationFunction(magnetization):
    '''
    Get the correlation function given a set of magnetisation values over different time values.

    Parameters:
    magnetisation : (numpy_array) Equally spaced magnetisation values.

    Returns: (numpy_array) The correlation function for all time differences, where dt=1.
    '''

    N = len(magnetization)

    correlation_function = np.zeros(N-1)

    for i in range(1, N):
        correlation_function[i-1] = (1/(N-i) * np.sum(magnetization[i:] * magnetization[:-i]) -
                                     1/(N-i)**2 * np.sum(magnetization[i:]) * np.sum(magnetization[:-i]))

    return correlation_function


def Mean(value_array, tau):
    return np.mean(value_array[::tau])


def Variance(value_array, tau):
    '''
    Take the variance over time steps 2*tau rather than successive time steps.

    Parameters:
    value_array : (array_like) array to take the variance over with equal time steps
    tau : (int) correlation time
    '''

    k_mean = np.mean(value_array[::tau])
    k_mean_squared = np.mean(value_array[::tau]**2)

    return np.sqrt(1 / (len(value_array) // (2*tau)) * (k_mean_squared - k_mean**2))


def Blockify(value_array, tau):
    '''
    For the magnetisation and the specific heat, the variance has to be calculated
    with blocks. This procedure splits the array into blocks and then calculates the mean and variance.

    value_array : (array_like) the array for which we'll estimate the variance
    tau : (int) correlation time
    '''

    N = len(value_array)
    blocked_array = value_array[:N * (N//(16 * tau))].reshape(N // (16*tau), 16*tau)
    values = np.zeros(N//(16*tau))
    for i, value_sequence in enumerate(blocked_array):
        values[i] = np.mean(value_sequence[::tau]**2) - np.mean(value_sequence[::tau])**2
    mean = np.mean(values)
    variance = np.sqrt(1 / (N // (16*tau)) * (np.mean(values**2) - np.mean(values)**2))

    return values, mean, variance


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


if __name__ == '__main__':
    N = 10
    T = 0.5

    tau_dict = {0.5: 44, 0.7: 76, 0.9: 59, 1.1: 25, 1.3: 10, 1.5: 14, 1.7: 5, 1.9: 6, 2.1: 5, 2.3: 2, 2.5: 2}
    tau = tau_dict[T]

    steps = 10 * tau

    spins, energy, magnetization = Animate(Aligned_spins(N), steps, T)



    # correlation_function = CorrelationFunction(magnetization)
    # tau = 0
    # for chi in correlation_function:
    #     if chi < 0:
    #         break
    #     tau += chi / correlation_function[0]
    #
    # print("T =", T, ", tau =", tau)
    # plt.plot(correlation_function)
    # plt.show()



    fig, ax = plt.subplots(2)
    fig.suptitle("T = " + str(round(T,1)))
    ax[0].plot(np.array(energy) / N**2)
    ax[0].title.set_text("Energy per unit")
    ax[0].set_ylim([-2,2])
    ax[1].plot(np.array(magnetization) / N**2)
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