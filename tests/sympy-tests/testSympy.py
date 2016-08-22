#!/usr/bin/env python
""" Test solving differential equations and systems of equations using SymPy.

    Tests  simpleOde and lorenz correspond with examples 3.1.5 and 3.1.6 from the
    OpenCOR tutorial, respectively.
    """

from sympy import *
from sympy.solvers.ode import (_undetermined_coefficients_match, checkodesol,
    classify_ode, classify_sysode, constant_renumber, constantsimp,
    homogeneous_order, infinitesimals, checkinfsol, checksysodesol)
from sympy.utilities.pytest import raises

import mpmath
from mpmath import odefun, nprint


def simpleOde():
    # dy/dt = -a*y + b
    print("\n\n========== Single equation simple ODE (3.1.5) ==========\n")
    t = Symbol("t")
    a = Symbol("a")
    b = Symbol("b")
    y = Function("y")
    eq = Eq(y(t).diff(t), -a * y(t) + b)
    print("The general solution for \n")
    pprint(eq)
    print("\nis:\n\n")
    solution = dsolve(eq, y(t))
    pprint(solution)

    print("\n\n ----- Adding algebraic complexity: ----------\n")
    # add complexity to this expression
    a = Function("a")
    b = Function("b")
    c = Symbol("c")

    a = b(c) + c**2
    b = 2.*c
    print("a = ")
    pprint(a)
    print("\nb = ")
    pprint(b)
    eq = Eq(y(t).diff(t), -a * y(t) + b)
    print("\nsolution is now:\n\n")
    solution = dsolve(eq, y(t))
    pprint(solution)
    #print(solution)

    print("\n\n ----- Set an algebraic initial value: ----------\n")
    # Set initial value for y as an algebraic expression
    y0 = Function("y0")

    y0 = 1./c**2
    print("\ny(0) = ")
    pprint(y0)

    print("\nsolution is now:\n\n")
    solution = dsolve(eq, y(t), "default", True, ics={y(0) : y0})
    pprint(solution)


def lorenz():
    print("\n\n\n\n========= Lorenz Equations (3.1.6) =========\n")

    t = Symbol("t")
    x = Function("x")
    y = Function("y")
    z = Function("z")

    sigma = Symbol("sigma")
    rho = Symbol("rho")
    beta = Symbol("beta")

    # Lorenz equations
    lorenzEqs = (
        Eq(x(t).diff(t), sigma * (y(t) - x(t))),
        Eq(y(t).diff(t), x(t) * (rho - z(t)) - y(t)),
        Eq(z(t).diff(t), x(t) * y(t) - beta * z(t))
    )

    # Parameter values
    sigma = 10.
    rho = 28.
    beta = 8./3.

    print("\nSystem info:")
    pprint(classify_sysode(lorenzEqs))


# Attempt to solve an underdetermined systems of equations.
def underdetermined():
    print("\n\n========== Underdetermined system (2 eq, 3 unknowns) ==========\n")
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    # no solution
    sysEq = (
        Eq(1, a + b + c),
        Eq(0, a + b + c)
    )
    print("\nThe solution for \n")
    pprint(sysEq)
    print("\nis (should have no solution):\n")
    solution = solve(sysEq, (a, b, c))
    pprint(solution)

    # infinite solutions
    sysEq = (
        Eq(1, a + b + c),
        Eq(0, a + b + 2.*c)
    )
    print("\nThe solution for \n")
    pprint(sysEq)
    print("\nis (should have infinite solutions):\n")
    solution = solve(sysEq, (a, b, c))
    pprint(solution)


# Attempt to solve an overdetermined systems of equations.
def overdetermined():
    print("\n\n========== Overdetermined system (3 eq, 2 unknowns) ==========\n")
    a = Symbol("a")
    b = Symbol("b")

    sysEq = (
        Eq(1, -a + b),
        Eq(-1, 2 * a + b),
        Eq(-2, -3 * a + b)
    )
    print("\nThe solution for \n")
    pprint(sysEq)
    print("\nis (should have no solution or inconsistent solutions):\n")
    solution = solve(sysEq, (a, b))
    pprint(solution)



if __name__ == "__main__":
    simpleOde()
    lorenz()

    underdetermined()
    overdetermined()
