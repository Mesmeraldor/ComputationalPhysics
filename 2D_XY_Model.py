import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def AlignedSpins(N):
    '''
    Returns a (N,N,2) numpy array corresponding to a configuration of aligned spins in a random direction

    Arguments:
        N : (int) number of spins along one direction. N*N being the total number of spins
    '''
    orientation = np.random.uniform(0, 2*np.pi)
    return np.ones((N,N,2))*[np.cos(orientation), np.sin(orientation)]


def RandomSpins(N):
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
    old_spins = AlignedSpins(N)
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


def ProbabilityOfChange(delta_e, T):
    '''
    Returns the probability of changing the configuration of one spin given the gain (or loss) of energy as a float

    Parameters:
        delta_e: (float) gain or loss of energy by rotating a spin
        T : (float) temperature of the system
    '''
    if delta_e <= 0:
        return 1
    else:
        return np.exp(-delta_e/T)
    

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
    energy -= np.sum(spins * magnetic_field[np.newaxis, np.newaxis, :])
    return energy


def UpdateSpin(spins, T, i, j, range=1.):
    '''
    Returns the new orientation of the spin at position [i,j] as a (2) numpy array,
    and the gain or loss of energy of this rotation as a float

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
        T: (float) temperature of the system
        i: (int) number of the row of the spin to change
        j: (int) number of the column of the spin to change
        range: (float) range of rotation for the spin (1 -> rotation up to 2*pi, 0 -> no rotation)
    '''
    orientation = range * np.random.uniform(-np.pi, np.pi)
    cosa, sina = spins[i][j][0], spins[i][j][1]
    cosb, sinb = np.cos(orientation), np.sin(orientation)
    new_spin = np.array([cosa*cosb - sina*sinb, sina*cosb + cosa*sinb])
    energy_before = - np.dot(spins[i][j], spins[(i+1)%N][j]) - np.dot(spins[i][j], spins[i][(j+1)%N]) - np.dot(spins[i][j], spins[(i-1)%N][j]) - np.dot(spins[i][j], spins[i][(j-1)%N])
    energy_after = - np.dot(new_spin, spins[(i+1)%N][j]) - np.dot(new_spin, spins[i][(j+1)%N]) - np.dot(new_spin, spins[(i-1)%N][j]) - np.dot(new_spin, spins[i][(j-1)%N])
    delta_e = energy_after - energy_before - np.dot(new_spin, magnetic_field) + np.dot(spins[i][j], magnetic_field)
    probability = ProbabilityOfChange(delta_e, T)
    cursor = np.random.uniform(0,1)
    if cursor < probability:
        return new_spin, delta_e
    else:
        return spins[i][j], 0
    

def Update(spins, T):
    '''
    Returns the new configuration of spins after one timestep as a (N,N,2) numpy array,
    and the gain or loss of energy of this change as a float

    Parameters:
        spins: ((N,N,2) numpy array) configuration of the spins [[[cos, sin],...],...]
        T: (float) temperature of the system
    '''
    N = len(spins)

    delta_e = 0
    for _ in range(N**2):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        spin, de = UpdateSpin(spins, T, i, j)
        spins[i][j] = spin
        delta_e += de
    return spins, delta_e


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
    elif steps == 0:
        spins, delta_e = Update(spins, T)
        energy.append(Energy(spins))
        magnetization.append(Magnetization(spins))
        c = 0
        while abs(energy[c+1]-energy[c]) > 0.01 and abs(magnetization[c+1]-magnetization[c]) > 0.01:
            for i in range(N**2):
                spins, delta_e = Update(spins, T)
                energy += [energy[i]+delta_e]
                magnetization += [Magnetization(spins)]
            c += 1
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
    return np.mean(value_array[::2*tau])


def StandardDeviation(value_array, tau):
    '''
    Take the variance over time steps 2*tau rather than successive time steps.

    Parameters:
    value_array : (array_like) array to take the variance over with equal time steps
    tau : (int) correlation time
    '''

    k_mean = np.mean(value_array[::2*tau])
    k_mean_squared = np.mean(value_array[::2*tau]**2)

    return np.sqrt(1 / (len(value_array) // (2 * tau) - 1) * (k_mean_squared - k_mean**2))


def Blockify(value_array, tau):
    '''
    For the magnetisation and the specific heat, the variance has to be calculated
    with blocks. This procedure splits the array into blocks and then calculates the mean and variance.

    value_array : (array_like) the array for which we'll estimate the variance
    tau : (int) correlation time
    '''

    N = len(value_array)
    blocked_array = value_array[:16 * tau * (N//(16 * tau))].reshape(N // (16*tau), 16*tau)
    values = np.zeros(N // (16*tau))
    for i, value_sequence in enumerate(blocked_array):
        values[i] = np.mean(np.square(value_sequence)) - np.mean(value_sequence)**2
    mean = np.mean(values)
    standard_deviation = np.sqrt(1 / (N // (16*tau) - 1) * (np.mean(values**2) - np.mean(values)**2))

    return values, mean, standard_deviation


def VectorTorgb(angle):
    '''
    Get the rgb value for the given `angle`

    Parameters:
    angle : (float) The angle in radians
    
    Returns: (array_like) The rgb value as a tuple with values [0...1]
    '''

    # normalize angle
    angle = angle % (2 * np.pi)

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 1, 1))


def PlotSpinConfiguration(N, T):
    '''
    Plots arrows corresponding to the spins in a N*N lattice
    '''
    spins, energy, magnetization = Animate(RandomSpins(N), steps, T)

    transition = np.reshape(spins, (N*N, 2))
    U = np.reshape(transition[:,0], (N,N))
    V = np.reshape(transition[:,1], (N,N))
    angles = np.arctan2(V, U)
    
    c3 = np.array(list(map(VectorTorgb, angles.flatten())))
    
    fig, ax = plt.subplots()
    q = ax.quiver(U, V, color=c3)
    plt.show()


def PlotEnergyMagnetization(N, T):
    '''
    plots the energy per spin vs the temperature and the magnetization per spin versus temperature
    '''
    spins, energy, magnetization = Animate(RandomSpins(N), steps, T)
    fig, ax = plt.subplots(2)
    fig.suptitle("T = " + str(round(T,1)))
    ax[0].plot(np.array(energy) / N**2)
    ax[0].title.set_text("Energy per spin")
    ax[0].set_ylim([-2,2])
    ax[1].plot(np.array(magnetization) / N**2)
    ax[1].title.set_text("Magnetization per spin")
    ax[1].set_ylim([-1,1])
    plt.tight_layout()
    plt.show()


def CorrelationTime(N, T):
    '''
    prints the correlation time given the size and temperature
    plots the correlation function
    '''
    spins, energy, magnetization = Animate(RandomSpins(N), steps, T)
    correlation_function = CorrelationFunction(magnetization)
    tau = 0
    for chi in correlation_function:
        if chi < 0:
            break
        tau += chi / correlation_function[0]
    
    print("T =", T, ", tau =", tau)
    plt.plot(correlation_function)
    plt.show()


def PhysicalQuantities(N, T):
    '''
    Prints the energy per spin, magnetization per spin,
    heat capacity and magnetic susceptibility of the system at a given temperature T
    '''
    spins, energy, magnetization = Animate(RandomSpins(N), steps, T)
    mean_energy = Mean(energy, tau)
    mean_magnetization = Mean(magnetization, tau)
    standard_deviation_energy = StandardDeviation(energy, tau)
    standard_deviation_magnetization = StandardDeviation(magnetization, tau)

    values, mean_heat_capacity, standard_deviation_heat_capacity = Blockify(energy, tau)
    values, mean_magnetic_susceptibility, standard_deviation_magnetic_susceptibility = Blockify(magnetization, tau)

    print()
    print('Temperature : ', T)
    print()
    print('mean energy : ', mean_energy / N / N , ', error energy : ' , 2 * standard_deviation_energy / N / N)
    print()
    print('mean magnetization : ' , mean_magnetization / N / N, ', error magnetization : ' , 2 * standard_deviation_magnetization / N / N)
    print()
    print('mean heat capacity : ' , mean_heat_capacity / T / T / N / N , ', error heat capacity : ' , 2 * standard_deviation_heat_capacity / T / T / N / N)
    print()
    print('mean magnetic susceptibility : ' , mean_magnetic_susceptibility / T / N / N , ', error magnetic_susceptibility : ' , 2 * standard_deviation_magnetic_susceptibility / T / N / N)
    print()


def FindVortices(spins):
    '''
    Find the coordinates where points circle about a particular point.

    Parameters:
    spins : (array_like) An (N,N,2) array of spin values.
    '''

    pos_coords = []
    neg_coords = []

    for i in range(N):
        for j in range(N):
            next_x = (j+1)%N
            next_y = (i+1)%N
            lb_angle = np.atan2(spins[i,j,1], spins[i,j,0])
            rb_angle = np.atan2(spins[i,next_x,1], spins[i,next_x,0])
            lt_angle = np.atan2(spins[next_y,j,1], spins[next_y,j,0])
            rt_angle = np.atan2(spins[next_y,next_x,1], spins[next_y,next_x,0])

            b_delta = (rb_angle - lb_angle + np.pi) % (2*np.pi) - np.pi
            r_delta = (rt_angle - rb_angle + np.pi) % (2*np.pi) - np.pi
            t_delta = (lt_angle - rt_angle + np.pi) % (2*np.pi) - np.pi
            l_delta = (lb_angle - lt_angle + np.pi) % (2*np.pi) - np.pi

            if b_delta + r_delta + t_delta + l_delta > 0.1:
                pos_coords += [(j+.5,i+.5)]
            if b_delta + r_delta + t_delta + l_delta < -0.1:
                neg_coords += [(j+.5, i+.5)]

    return pos_coords, neg_coords


def SpinPlotter(spins):
    '''
    Plot the curl and the states of the spin in one figure.

    Parameters:
    spins : (array_like) An (N,N,2) array of the spin values.
    '''

    fig, ax = plt.subplots()

    transition = np.reshape(spins, (N*N, 2))
    U = np.reshape(transition[:,0], (N,N))
    V = np.reshape(transition[:,1], (N,N))
    angles = np.arctan2(V, U)

    c3 = np.array(list(map(VectorTorgb, angles.flatten())))

    q = ax.quiver(U, V, color=c3, scale=100)
    ax.set_box_aspect(1)
    ax.set_title("T="+ str(T))

    pos_poles, neg_poles = FindVortices(spins)

    for coord in pos_poles:
        ax.add_patch(plt.Circle(coord, 1, color='r', fill=False))
    for coord in neg_poles:
        ax.add_patch(plt.Circle(coord, 1, color='b', fill=False))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    N = 50
    T = 2.5
    h = 0.5
    magnetic_field = np.array([0,h])

    tau_dict = {0.5: 44, 0.7: 76, 0.9: 59, 1.1: 25, 1.3: 10, 1.5: 14, 1.7: 5, 1.9: 6, 2.1: 5, 2.3: 2, 2.5: 2}
    tau_dict_ext_field = {0.5: 10, 0.7: 10, 0.9: 10, 1.1: 10, 1.3: 9, 1.5: 8, 1.7: 12, 1.9: 7, 2.1: 5, 2.3: 5, 2.5: 10}
    if h == 0:
        tau = tau_dict[T]
    else:
        tau = tau_dict_ext_field[T]

    steps = 100