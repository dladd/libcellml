#!/usr/bin/env python
""" Test solving differential equations and systems of equations using SymPy.

    Tests  simpleOde and lorenz correspond with examples 3.1.5 and 3.1.6 from the
    OpenCOR tutorial, respectively.
    """

from sympy import (Symbol, Function, Eq, exp, solve, dsolve, simplify, expand, factor, cancel, pprint)
from sympy.solvers.ode import (classify_ode, classify_sysode, checksysodesol)

def simpleOde():
    # dy/dt = -a*y + b
    print("\n\n========== Single equation simple ODE (3.1.5) ==========\n")
    t = Symbol("t")
    a = Symbol("a")
    b = Symbol("b")
    y0 = Symbol("y0")
    y = Function("y")

    eq = Eq(y(t).diff(t), -a * y(t) + b)
    print("The general solution for \n")
    pprint(eq)
    print("\nis:\n")
    solution = dsolve(eq, y(t), "default", True, ics={y(0): y0})
    pprint(solution)
    # Print some simplified versions of the expression
    print("Which may also be expressed: \n")
    pprint(expand(solution))
    pprint(factor(solution))
    pprint(cancel(solution))

    print("\n\n\n ----- Adding algebraic complexity: ----------\n")
    # add complexity to this expression
    a = Function("a")
    b = Function("b")
    c = Symbol("c")

    a = 1/c**2
    b = 4*c + 1
    print("a = ")
    pprint(a)
    print("\nb = ")
    pprint(b)
    eq = Eq(y(t).diff(t), -a * y(t) + b)
    print("\nsolution is now:\n\n")
    solution = dsolve(eq, y(t), "default", True, ics={y(0): y0})
    pprint(solution)
    # Print some simplified versions of the expression
    print("Which may also be expressed: \n")
    pprint(expand(solution))
    pprint(factor(solution))
    pprint(cancel(solution))

    print("\n\n ----- Set an algebraic initial value: ----------\n")
    # Set initial value for y as an algebraic expression
    y0 = Function("y0")

    y0 = b/a
    print("\ny(0) = ")
    pprint(y0)

    print("\nsolution is now:\n\n")
    solution = dsolve(eq, y(t), "default", True, ics={y(0) : y0})
    pprint(simplify(solution))
    # Print some simplified versions of the expression
    print("Which may also be expressed: \n")
    pprint(expand(solution))
    pprint(factor(solution))
    pprint(cancel(solution))

    print("\n\n ----- Check units consistency: ----------\n")
    t = Symbol("t")
    a = Symbol("a")
    b = Symbol("b")
    y0 = Symbol("y0")
    y = Function("y")

    ut = Symbol("second")
    ua = Symbol("perSecond")
    ub = Symbol("meterPerSecond")
    uy0 = Symbol("meter")
    uy = Symbol("meter")

    subMap = {
        t: ut,
        a: ua,
        b: ub,
        y0: uy0,
        y(t): uy
    }

    eq = Eq(y(t), b/a + (y0 - b/a) * exp(-a*t))
    print("Equation: \n")
    pprint(eq)

    print("\nUnits substitution: \n")
    eqUnits = eq.subs(subMap)
    pprint(eqUnits)

    print("\nWhich reduces to: \n")
    s = Symbol("second")
    m = Symbol("meter")

    ut = s
    ua = 1/s
    ub = m/s
    uy0 = m
    uy = m

    subMap = {
        t: ut,
        a: ua,
        b: ub,
        y0: uy0,
        y(t): uy
    }

    eqUnits = eq.subs(subMap)
    pprint(eqUnits)


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

# Attempt to solve the Rivlin-Saunders (1951) system for stress and strain.
# See: https://models.physiomeproject.org/exposure/8ed2abe0b72de8d146522d8331999c42/rivlin_saunders_1951.cellml/view
def rivlin():
    print("\n\n========== Mooney-Rivlin system ==========\n")
    E11 = Symbol("E11")
    E22 = Symbol("E22")
    E33 = Symbol("E33")
    E12 = Symbol("E12")
    E13 = Symbol("E13")
    E23 = Symbol("E23")

    Tdev11 = Symbol("Tdev11")
    Tdev22 = Symbol("Tdev22")
    Tdev33 = Symbol("Tdev33")
    Tdev12 = Symbol("Tdev12")
    Tdev13 = Symbol("Tdev13")
    Tdev23 = Symbol("Tdev23")

    c1 = Symbol("c1")
    c2 = Symbol("c2")

    sysEq = (
        Eq(Tdev11, 2.0 * c1 + 4.0 * c2 * (E22 + E33) + 4 * c2),
        Eq(Tdev22, 2.0 * c1 + 4.0 * c2 * (E11 + E33) + 4 * c2),
        Eq(Tdev33, 2.0 * c1 + 4.0 * c2 * (E11 + E22) + 4 * c2),
        Eq(Tdev12, -4.0 * E12 * c2),
        Eq(Tdev13, -4.0 * E13 * c2),
        Eq(Tdev23, -4.0 * E23 * c2)
    )

    print("\nThe system of equations\n")
    pprint(sysEq)

    print("\n\n May be solved for stress:\n")
    stressSolution = solve(sysEq, (Tdev11, Tdev22, Tdev33, Tdev12, Tdev13, Tdev23))
    pprint(stressSolution)

    print("\n\n or strain:\n")
    strainSolution = solve(sysEq, (E11, E22, E33, E12, E13, E23))
    pprint(strainSolution)


# Attempt to solve the guccione system for stress and strain.
def guccione():
    print("\n\n========== Guccione system (models.physiomeproject.org/e/26d) ==========\n")
    E11 = Symbol("E11")
    E22 = Symbol("E22")
    E33 = Symbol("E33")
    E12 = Symbol("E12")
    E13 = Symbol("E13")
    E23 = Symbol("E23")

    c1 = Symbol("c1")
    c2 = Symbol("c2")
    c3 = Symbol("c3")
    c4 = Symbol("c4")
    c5 = Symbol("c5")

    Tdev11 = Symbol("Tdev11")
    Tdev22 = Symbol("Tdev22")
    Tdev33 = Symbol("Tdev33")
    Tdev12 = Symbol("Tdev12")
    Tdev13 = Symbol("Tdev13")
    Tdev23 = Symbol("Tdev23")

    Q = Symbol("Q")

    sysEq = (
        Eq(Q, 2 * c2 * (E11 + E22 + E33) +
           c3 * E11 ** 2 +
           c4 * (E33 ** 2 + E22 ** 2 + 2 * E23 ** 2) +
           2 * c5 * (E13 ** 2 + E12 ** 2)),
        Eq(Tdev11, c1 * exp(Q) * (c2 + c3 * E11)),
        Eq(Tdev22, c1 * exp(Q) * (c2 + c4 * E22)),
        Eq(Tdev33, c1 * exp(Q) * (c2 + c4 * E33)),
        Eq(Tdev12, c1 * exp(Q) * c5 * E12),
        Eq(Tdev13, c1 * exp(Q) * c5 * E13),
        Eq(Tdev23, c1 * exp(Q) * c4 * E23)
    )

    print("\nThe system of equations\n")
    pprint(sysEq)

    print("\n\n May be solved for stress:\n")
    stressSolution = solve(sysEq, (Tdev11, Tdev22, Tdev33, Tdev12, Tdev13, Tdev23))
    pprint(stressSolution)

    print("\n\n but not strain:\n")
    strainSolution = solve(sysEq, (E11, E22, E33, E12, E13, E23))
    pprint(strainSolution)

    print("\n\n Try declaring Q as a separate equation and defining constants:\n")
    Q = (2 * c2 * (E11 + E22 + E33) +
        c3 * E11 ** 2 +
        c4 * (E33 ** 2 + E22 ** 2 + 2 * E23 ** 2) +
        2 * c5 * (E13 ** 2 + E12 ** 2))
    c1 = 0.88
    c2 = 0
    c3 = 18.5
    c4 = 3.58
    c5 = 3.26

    sysEq = (
        Eq(Tdev11, c1 * exp(Q) * (c2 + c3 * E11)),
        Eq(Tdev22, c1 * exp(Q) * (c2 + c4 * E22)),
        Eq(Tdev33, c1 * exp(Q) * (c2 + c4 * E33)),
        Eq(Tdev12, c1 * exp(Q) * c5 * E12),
        Eq(Tdev13, c1 * exp(Q) * c5 * E13),
        Eq(Tdev23, c1 * exp(Q) * c4 * E23)
    )

    print("\n\n which then gives stress:\n")
    stressSolution = solve(sysEq, (Tdev11, Tdev22, Tdev33, Tdev12, Tdev13, Tdev23))
    pprint(stressSolution)

    print("\n\n And strain as:\n")
    strainSolution = solve(sysEq, (E11, E22, E33, E12, E13, E23))
    pprint(strainSolution)


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
    rivlin()

    underdetermined()
    overdetermined()

    guccione()