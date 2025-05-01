##### A study of the 2D XY-model using Monte-Carlo simulation by Joran and Gabriel #####

### Description ###

We aim at giving some figures for relevant physical quantities such as the energy per spin, magnetization per spin, heat capacity and magnetic susceptibility. The study is broaden by looking at vortices creation for low temperature and the addition of an external magnetic field.

### Variables ###

The variables for the simulation are:
    - T: The equilibrium temperature of the system in units of 1/k_b. We recommend using values in [0.5, 2.5]
    - N: The number of spins along one direction. The spins forms a square lattice of dimensions N*N. We recommend using N = 50
    - h: The field coupling strength. We recommend using 0 to turn off the external magnetic field or using 0.5
    - steps: The number of steps (e.g. time) for which you want to run the simulation. To reach equilibrium with the settings above we recommend 100 but for other purposes the recommended number of steps will be explained later on.

### How to use the code? ###

The code enables you to visualize and compute different configurations and quantities.

    - To visualize a configuration of spins: In "if __name__ == '__main__':" in the file Monte_carlo_simulation.py run the function PlotSpinConfiguration(N, T, steps). We recommend using steps = 100

    - To plot the energy per spin and magnetization per spin with respect to time: In "if __name__ == '__main__':" in the file Monte_carlo_simulation.py run the function PlotEnergyMagnetization(N, T, steps). We recommend using steps = 100.

    - To get the correlation time and to plot the correlation function: In "if __name__ == '__main__':" in the file Monte_carlo_simulation.py run the function CorrelationTime(N, T, steps). We recommend using steps = 100.

    - To get the values and errorbars of the energy per spin, the magnetization per spin, the heat capacity, the magnetic susceptibility: In "if __name__ == '__main__':" in the file Monte_carlo_simulation.py run the function PhysicalQuantities(N, T, steps). We recommend using steps = 160 * tau. 

    - To plot a configuration of spins with vortices outlined: In "if __name__ == '__main__':" in the file Monte_carlo_simulation.py run the function SpinPlotter(N, T, steps). We recommend using steps = 100

