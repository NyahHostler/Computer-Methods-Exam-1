# region imports
import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt
# endregion

# region function definitions
def S(x):
    """
    Evaluates the integral S(x) = ∫ sin(t²) dt from 0 to x
    using the quad integration routine from scipy.integrate.
    parameter: x – upper bound of the integral
    returns the computed integral value
    AI assisted in writing this function
    """
    s = quad(lambda t: np.sin(t**2), 0, x)  # numerically integrate sin(t²) between 0 and x
    return s[0]

def Exact(x):
    """
    Returns the analytical solution of the IVP at a specific x value.
    Exact form: y = 1 / (2.5 - S(x)) + 0.01*x²
    parameter: x where the solution is evaluated
    returns the corresponding y value
    """
    return 1 / (2.5 - S(x)) + 0.01 * x**2

def ODE_System(x, y):
    """
    Represents the differential equation used by the solver:
    y' = (y - 0.01x²)² sin(x²) + 0.02x
    parameters:
    x : independent variable
    y : list containing the state variable [y]
    returns a list containing the derivative [y']
    AI assisted with this implementation
    """
    Y = y[0]
    Ydot = (Y - 0.01*x**2)**2 * np.sin(x**2) + 0.02*x  # evaluate derivative for the current state
    return [Ydot]

def Plot_Result(*args):
    """
    Displays a graph comparing the numerical IVP solution
    with the analytical solution using the required formatting.
    arguments: (xRange_Num, y_Num, xRange_Xct, y_Xct)
    returns: none
    """
    xRange_Num, y_Num, xRange_Xct, y_Xct = args  # extract arrays for numerical and exact solutions

    fig, ax = plt.subplots()

    # draw exact solution with a line and numerical points with triangle markers
    ax.plot(xRange_Xct, y_Xct, '-', label='Exact')
    ax.plot(xRange_Num, y_Num, '^', label='Numerical')

    # configure x-axis limits and tick style
    ax.set_xlim(0.0, 6.0)
    ax.set_xlabel('x')
    ax.tick_params(axis='x', direction='in', top=True, bottom=True)

    # configure y-axis limits and tick style
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('y')
    ax.tick_params(axis='y', direction='in', left=True, right=True)

    # display axis tick labels with one decimal of precision
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # include legend and descriptive title
    ax.legend()
    ax.set_title("IVP: y' = (y - 0.01x\u00b2)\u00b2sin(x\u00b2) + 0.02x, y(0) = 0.4")

    plt.tight_layout()
    plt.show()

def main():
    """
    Solves the initial value problem:
    y'=(y-0.01x**2)**2*sin(x**2)+0.02x
    y(0)=0.4
    Both the numerical approximation and the exact solution
    are then plotted using the specified formatting.
    Procedure:
    1. Define x values for numerical and exact evaluation
    2. Apply solve_ivp to compute the numerical solution
    3. Evaluate the analytical expression across the domain
    4. Plot both curves together for comparison
    """
    # Step 1: define x domains
    xRange = np.arange(0, 5.2, 0.2)          # grid used for the numerical method (step size = 0.2)
    xRange_xct = np.linspace(0, 5, 500)      # dense grid for smoother exact solution curve

    # Step 2: numerically solve the IVP
    Y0 = [0.4]                                # initial value y(0) = 0.4
    sln = solve_ivp(ODE_System, [0, 5], Y0, t_eval=xRange)   # integrate using default RK45 solver

    # Step 3: evaluate the exact solution along the continuous x range
    xctSln = np.array([Exact(x) for x in xRange_xct])

    # Step 4: generate the comparison plot
    Plot_Result(xRange, sln.y[0], xRange_xct, xctSln)
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion