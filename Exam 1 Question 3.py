# region imports
from scipy.integrate import solve_ivp
from math import sin
import math
import numpy as np
from matplotlib import pyplot as plt
# AI assistance (Claude) was used while developing this file
# endregion

# region class definitions
class circuit():
    def __init__(self, R=10, L=20, C=0.05, A=20, w=20, p=0):
        """
        Initializes an object representing an RLC circuit.
        :param R: electrical resistance (Ohms)
        :param L: inductance value (Henrys)
        :param C: capacitance value (Farads)
        :param A: magnitude of the sinusoidal voltage source (Volts)
        :param w: angular frequency of the source (rad/s)
        :param p: phase shift of the voltage input (radians)
        """
        # region attributes
        self.R = R
        self.L = L
        self.C = C
        self.A = A
        self.w = w
        self.p = p
        self.t = None       # array storing simulation time values
        self.i_L = None     # inductor current results
        self.vc = None      # capacitor voltage results
        self.i1 = None      # current flowing through resistor
        self.i2 = None      # current flowing through capacitor
        # endregion

    # region methods
    def ode_system(self, t, X):
        """
        Differential equation callback used by solve_ivp.
        State vector definition:
        X[0] -> capacitor voltage (vc)
        X[1] -> inductor current (i_L)
        :param t: current simulation time
        :param X: list containing state variables [vc, i_L]
        :return: derivatives [dvc/dt, di_L/dt]
        AI assistance used
        """
        vc = X[0]
        i_L = X[1]

        v = self.A * sin(self.w * t + self.p)   # instantaneous voltage source value

        dvc_dt = (i_L - vc / self.R) / self.C
        diL_dt = (v - vc) / self.L

        return [dvc_dt, diL_dt]

    def simulate(self, t=10, pts=500):
        """
        Runs a transient simulation of the circuit dynamics using solve_ivp.
        :param t: duration of the simulation in seconds
        :param pts: number of time points to evaluate
        :return: nothing; results are stored in class attributes
        """
        # Step 1: define time grid and starting conditions
        t_eval = np.linspace(0, t, pts)
        X0 = [0, 0]  # starting values: vc(0)=0 and i_L(0)=0

        # Step 2: numerically integrate the ODE system
        sln = solve_ivp(self.ode_system, [0, t], X0, t_eval=t_eval)

        # Step 3: save the computed data into the object
        self.t = sln.t
        self.vc = sln.y[0]
        self.i_L = sln.y[1]
        self.i1 = self.vc / self.R          # resistor current
        self.i2 = self.i_L - self.i1        # capacitor current

    def doPlot(self, ax=None):
        """
        Generates a plot of i1, i2, and vc as functions of time.
        Currents share the left axis, while capacitor voltage
        is plotted on a secondary right axis.
        :param ax: existing matplotlib axis (optional)
        :return: none; displays figure
        AI assistance used
        """
        if ax is None:
            fig, ax = plt.subplots()
            QTPlotting = False
        else:
            QTPlotting = True
        ax2 = ax.twinx()

        l1, = ax.plot(self.t, self.i1, 'k-', label='$i_1(t)$')
        l2, = ax.plot(self.t, self.i2, 'k--', label='$i_2(t)$')
        l3, = ax2.plot(self.t, self.vc, 'k:', label='$v_c(t)$')

        # configure left y-axis for current values
        ax.set_xlabel('t (s)')
        ax.set_ylabel('$i_1, i_2$(A)')
        ax.tick_params(axis='x', direction='in', top=True, bottom=True)
        ax.tick_params(axis='y', direction='in', left=True, right=False)

        # configure right y-axis for capacitor voltage
        ax2.set_ylabel('$v_c(t)$(V)')
        ax2.tick_params(axis='y', direction='in', left=False, right=True)

        # legend and descriptive title
        ax.legend(handles=[l1, l2, l3], loc='upper right')
        ax.set_title('RLC Circuit Response\nR={} Ω, L={} H, C={} F, v(t)={}·sin({}·t+{})'.format(
            self.R, self.L, self.C, self.A, self.w, self.p))

        ax.grid(True)

        if not QTPlotting:
            plt.show()

# endregion

# region function definitions
def main():
    """
    Driver routine used for solving the exam problem.
    Requests circuit parameters from the user and then
    performs the RLC simulation and plotting.
    :return: nothing
    """
    goAgain = True
    Circuit = circuit(R=10, L=20, C=0.05, A=20, w=20, p=0)  # instantiate circuit with default parameters

    while goAgain:
        # Step 1: prompt user to modify circuit parameters
        print("\nEnter circuit parameters (press Enter to keep current value):")

        st = input(f'R - Resistance (Ohms)? ({Circuit.R}): ').strip()
        Circuit.R = Circuit.R if st == '' else float(st)

        st = input(f'L - Inductance (Henrys)? ({Circuit.L}): ').strip()
        Circuit.L = Circuit.L if st == '' else float(st)

        st = input(f'C - Capacitance (Farads)? ({Circuit.C}): ').strip()
        Circuit.C = Circuit.C if st == '' else float(st)

        st = input(f'A - Voltage amplitude (Volts)? ({Circuit.A}): ').strip()
        Circuit.A = Circuit.A if st == '' else float(st)

        st = input(f'w - Voltage frequency (rad/s)? ({Circuit.w}): ').strip()
        Circuit.w = Circuit.w if st == '' else float(st)

        st = input(f'p - Voltage phase (radians)? ({Circuit.p}): ').strip()
        Circuit.p = Circuit.p if st == '' else float(st)

        # Step 2: run simulation and generate plot
        Circuit.simulate(t=10, pts=500)
        Circuit.doPlot()

        # Step 3: prompt user to repeat simulation if desired
        goAgain = input('\nGo again? (No): ').strip().lower().__contains__('y')

# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion