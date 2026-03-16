# region imports
import math
# endregion

# region function definitions
def ode_system(x, y1, y2):
    """
    Defines the pair of first-order equations corresponding to
    the second-order differential equation y'' - y = x.

    State representation:
    y1 = y
    y2 = y'

    Therefore:
    y1' = y2
    y2' = y1 + x

    :param x: independent variable
    :param y1: value of y
    :param y2: value of y'
    :return: derivatives (dy1/dx, dy2/dx)
    AI assisted
    """
    dy1 = y2          # derivative of y1 with respect to x
    dy2 = y1 + x      # from rearranging y'' - y = x → y'' = y + x
    return dy1, dy2

def improved_euler(f, x0, y1_0, y2_0, h, x_end):
    """
    Numerically integrates a system of ODEs using the
    Improved Euler (Heun's) method.

    :param f: function representing the system f(x, y1, y2)
    :param x0: starting x value
    :param y1_0: initial y value
    :param y2_0: initial derivative y'
    :param h: integration step size
    :param x_end: x location where the solution is desired
    :return: final (y, y') values at x_end
    AI assisted
    """
    x = x0
    y1 = y1_0
    y2 = y2_0

    while round(x, 10) < round(x_end, 10):
        # predictor step using standard Euler approximation
        dy1, dy2 = f(x, y1, y2)
        y1_pred = y1 + h * dy1
        y2_pred = y2 + h * dy2

        # corrector step using average slope
        dy1_pred, dy2_pred = f(x + h, y1_pred, y2_pred)
        y1 = y1 + h * (dy1 + dy1_pred) / 2
        y2 = y2 + h * (dy2 + dy2_pred) / 2
        x += h

    return y1, y2

def runge_kutta(f, x0, y1_0, y2_0, h, x_end):
    """
    Computes the solution to the ODE system using
    the classical fourth-order Runge-Kutta method.

    :param f: function describing the ODE system
    :param x0: initial value of x
    :param y1_0: starting value of y
    :param y2_0: starting value of y'
    :param h: step size for integration
    :param x_end: point where the solution is evaluated
    :return: (y, y') evaluated at x_end
    AI helped with conceptualization
    """
    x = x0
    y1 = y1_0
    y2 = y2_0

    while round(x, 10) < round(x_end, 10):
        # evaluate the four RK4 slope estimates
        a1, a2 = f(x, y1, y2)
        b1, b2 = f(x + h/2, y1 + h*a1/2, y2 + h*a2/2)
        c1, c2 = f(x + h/2, y1 + h*b1/2, y2 + h*b2/2)
        d1, d2 = f(x + h,   y1 + h*c1,   y2 + h*c2)

        # combine slopes to update solution
        y1 = y1 + h * (a1 + 2*b1 + 2*c1 + d1) / 6
        y2 = y2 + h * (a2 + 2*b2 + 2*c2 + d2) / 6
        x += h

    return y1, y2

def exact(x):
    """
    Analytical solution of the differential equation
    y'' - y = x with initial conditions y(0)=1 and y'(0)=-2.

    Solution expressions:
    y = -x + cosh(x) - sinh(x)
    y' = -1 + sinh(x) - cosh(x)

    :param x: value where solution is evaluated
    :return: tuple containing (y, y')
    """
    y = -x + math.cosh(x) - math.sinh(x)
    dy = -1 + math.sinh(x) - math.cosh(x)
    return y, dy


def main():
    """
    Program to compute the solution of y'' - y = x
    using two numerical approaches: Improved Euler
    and the fourth-order Runge-Kutta method.

    The user provides the initial conditions, step size,
    and the x value where the solution should be evaluated.
    Results for y and y' are printed for both methods.
    """
    print("For the initial value problem y'' - y = x")

    goAgain = True
    # Step 1: obtain initial conditions and step size from the user
    y0 = float(input("Enter the value of y at x=0: "))
    dy0 = float(input("Enter the value of y' at x=0: "))
    h = float(input("Enter the step size for the numerical solution: "))

    while goAgain:
        # Step 2: request the x value where the solution is needed
        x_end = float(input("At what value of x do you want to know y and y'? "))

        # Step 3: compute solutions using both numerical methods
        y1_ie, y2_ie = improved_euler(ode_system, 0, y0, dy0, h, x_end)
        y1_rk, y2_rk = runge_kutta(ode_system, 0, y0, dy0, h, x_end)

        # Step 4: display computed results
        print(f"\nAt x={x_end:.3f}")
        print(f"For the improved Euler method: y={y1_ie:.3f}, and y'={y2_ie:.3f}")
        print(f"For the Runge-Kutta method: y={y1_rk:.3f}, and y'={y2_rk:.3f}")

        # Step 5: ask if another x value should be evaluated
        goAgain = input("\nDo you want to compute at a different x? (Y/N): ").strip().lower() == 'y'

# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion