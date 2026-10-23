"""physverify spec for the projectile engines in src/.

    physverify run tests/verify_projectile.py

Equations, written from Newton's second law for a ball launched from the
origin, state z = [x, y, vx, vy], g = 9.81 m/s^2 downward:

    vacuum:     dv/dt = (0, -g)
    linear:     dv/dt = (0, -g) - (b / m) v
    quadratic:  dv/dt = (0, -g) - (rho C_d A / (2 m)) |v| v,   A = pi r^2

Vacuum and linear drag have exact solutions (below); quadratic drag is
compared with an independent RK4 solution of the equation above.

Drag constants: v2 (ProjectileV2) uses b = 0.01 kg/s, a fixed value in the
code; v1 (Projectile) uses Stokes' law b = 6 pi eta r with eta = 1.81e-5 Pa s,
as the README describes. The spec takes each version's b as given.

What is checked (ball of m = 0.45 kg, r = 0.11 m, v0 = 30 m/s at 45 degrees):
- v2 RK4: order 4 for linear and quadratic drag at t = 1 s; exact (to
  round-off) in vacuum, where the solution is a quadratic in t.
- v2 Euler: order 1 for vacuum, linear and quadratic drag.
- v2 landing range in vacuum against v0^2 sin(2 angle) / g, at order 2:
  the landing point is interpolated linearly between the last two steps,
  which limits the range (not the trajectory) to second order.
- v1 Euler positions: order 1 in vacuum and with Stokes drag.

Not checked: the code has no time argument or source-term hook, so there is no
manufactured-solution check here; v1's landing range, which is the last grid
point above ground and not a smooth function of dt.
"""

import math
from pathlib import Path

import numpy as np

from physverify import Check, check_exact, check_order, reference_study, zero_source
from physverify.loaders import load_module_from_path

SRC = Path(__file__).resolve().parents[1] / "src"
STUBS = ("matplotlib", "matplotlib.pyplot")

G, RHO_AIR, CD = 9.81, 1.225, 0.47
MASS, RADIUS = 0.45, 0.11
V0, ANGLE = 30.0, 45.0
VX0, VY0 = V0 * math.cos(math.radians(ANGLE)), V0 * math.sin(math.radians(ANGLE))
Y0 = np.array([0.0, 0.0, VX0, VY0])
B_V2 = 0.01
B_V1 = 6.0 * math.pi * 1.81e-5 * RADIUS
T_END = 1.0
NS = {"rk4": (10, 20, 40, 80), "euler": (100, 200, 400, 800)}
ORDER = {"rk4": 4.0, "euler": 1.0}
# Linear drag is weak here (b / m = 0.022 1/s), so RK4 reaches round-off by 40 steps: use coarser ones.
NS_LINEAR = {"rk4": (2, 4, 8, 16), "euler": NS["euler"]}


def v2_module():
    return load_module_from_path("projectile_sim_v2_under_test", SRC / "projectile_sim_v2.py", stubs=STUBS)


def v1_module():
    return load_module_from_path("projectile_sim_v1_under_test", SRC / "projectile_sim_v1.py", stubs=STUBS)


def exact(mode, t, b=B_V2):
    """Closed-form state at time t for vacuum and linear drag."""
    if mode == "vacuum":
        return np.array([VX0 * t, VY0 * t - 0.5 * G * t * t, VX0, VY0 - G * t])
    k = b / MASS
    decay = math.exp(-k * t)
    return np.array([
        VX0 * (1 - decay) / k,
        (VY0 + G / k) * (1 - decay) / k - G * t / k,
        VX0 * decay,
        (VY0 + G / k) * decay - G / k,
    ])


def quadratic_rhs(_t, z):
    c = RHO_AIR * CD * math.pi * RADIUS**2 / (2 * MASS)
    speed = math.hypot(z[2], z[3])
    return np.array([z[2], z[3], -c * speed * z[2], -G - c * speed * z[3]])


def v2_integrate(mode, method):
    """State after n fixed steps of ProjectileV2.simulate (row n is at t = n dt while in flight)."""
    module = v2_module()

    def integrate(source, y0, t_end, n):
        if source is not zero_source or not np.array_equal(y0, Y0):
            raise ValueError("ProjectileV2 has no source hook and launches from (V0, ANGLE)")
        sim = module.ProjectileV2(mass=MASS, radius=RADIUS, C_d=CD)
        traj = sim.simulate(V0, ANGLE, mode=mode, dt=t_end / n, method=method)
        if len(traj) <= n + 1:
            raise ValueError("the ball landed before t_end")
        return traj[n]

    return integrate


def exact_study(integrate, mode, ns, order, b=B_V2):
    hs, errs = [], []
    ref = exact(mode, T_END, b)
    for n in ns:
        hs.append(T_END / n)
        errs.append(float(np.max(np.abs(integrate(zero_source, Y0, T_END, n) - ref))))
    if order is None:
        return check_exact(hs, errs, scale=V0)
    return check_order(hs, errs, order, scale=V0)


def vacuum_range():
    module = v2_module()
    sim = module.ProjectileV2(mass=MASS, radius=RADIUS, C_d=CD)
    exact_range = V0**2 * math.sin(math.radians(2 * ANGLE)) / G
    hs = [0.02, 0.01, 0.005, 0.0025]
    errs = [abs(sim.simulate(V0, ANGLE, mode="vacuum", dt=h, method="rk4")[-1, 0] - exact_range) for h in hs]
    return check_order(hs, errs, 2.0, scale=exact_range)


def v1_positions(mode):
    module = v1_module()
    b = B_V1 if mode == "linear" else 0.0

    def run():
        hs, errs = [], []
        ref = exact(mode, T_END, b)[:2]
        for n in NS["euler"]:
            sim = module.Projectile(MASS, RADIUS)
            _, xs, ys = sim.simulate(V0, ANGLE, mode=mode, dt=T_END / n)
            hs.append(T_END / n)
            errs.append(float(np.max(np.abs(np.array([xs[n], ys[n]]) - ref))))
        return check_order(hs, errs, 1.0, scale=V0)

    return run


def checks():
    out = []
    for method in ("rk4", "euler"):
        ns, order = NS[method], ORDER[method]
        vac_order = None if method == "rk4" else order
        family = "patch" if method == "rk4" else "reference"
        out += [
            Check(family, f"v2 {method} vacuum vs exact solution",
                  lambda m=method, ns=ns, o=vac_order: exact_study(v2_integrate("vacuum", m), "vacuum", ns, o)),
            Check("reference", f"v2 {method} linear drag vs exact solution",
                  lambda m=method, o=order: exact_study(v2_integrate("linear", m), "linear", NS_LINEAR[m], o)),
            Check("reference", f"v2 {method} quadratic drag",
                  lambda m=method, ns=ns, o=order: reference_study(v2_integrate("quadratic", m), quadratic_rhs,
                                                                   Y0, T_END, ns, o)),
        ]
    out += [
        Check("limit", "v2 rk4 vacuum range vs v0^2 sin(2 angle) / g (order 2)", vacuum_range),
        Check("reference", "v1 euler vacuum positions", v1_positions("vacuum")),
        Check("reference", "v1 euler Stokes-drag positions", v1_positions("linear")),
    ]
    return out
