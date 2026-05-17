"""
Cross-Validation Verification System (Quality Audit Utility).

This utility module computes explicit position metric residuals comparing manual
vectorized RK4 operations against adaptive high-order Dormand-Prince (RK45)
reference paths sourced from scipy.integrate. It uses cubic spline transformations
to map mismatched spatial matrices onto identical grid timelines.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    from projectile_sim_v2 import ProjectileV2
    print("Verification System: Target state-space structures verified.")
except ImportError:
    print("Verification Failure: Target module dependency unresolvable.")
    exit()


def ground_event(_t, state):
    """
    Isolate zero-plane terrain coordinate cross boundaries.

    Acts as a terminal conditional break flag for SciPy step processors.

    Parameters:
    -----------
    _t : float
        Independent integration continuous time dimension variable (s).
    state : ndarray, shape (4,)
        Current vector array state parameters tracking [x, y, vx, vy]^T.

    Returns:
    --------
    y_coordinate : float
        Vertical altitude position value (m); zero signals terrain intersection.
    """
    return state[1]


ground_event.terminal = True
ground_event.direction = -1


def run_scipy_audit():
    """
    Execute spatial track convergence analysis across independent integrators.

    Computes horizontal and vertical trajectory matrices, normalizes non-uniform
    step spans via cubic splines, isolates divergence residuals, and saves
    diagnostic error metrics.
    """
    sim = ProjectileV2(mass=0.45, radius=0.11)
    v0, angle = 30.0, 45.0
    dt = 0.01

    print("Initializing execution convergence audit...")

    rk4_data = sim.simulate(v0, angle, mode="quadratic", dt=dt, method="rk4")
    t_rk4 = np.arange(len(rk4_data)) * dt

    rad = np.radians(angle)
    initial_state = [0.0, 0.0, v0 * np.cos(rad), v0 * np.sin(rad)]
    t_span = (0.0, 10.0)

    def scipy_derivs(_t, z):
        return sim._state_derivative(z, mode="quadratic")  # pylint: disable=protected-access

    sol = solve_ivp(
        scipy_derivs,
        t_span,
        initial_state,
        method="RK45",
        events=ground_event,
        rtol=1e-8,
        atol=1e-8,
    )

    max_common_t = min(t_rk4[-1], sol.t[-1])
    comparison_time = np.arange(0, max_common_t, dt)

    sci_x_interp = interp1d(sol.t, sol.y[0], kind="cubic")(comparison_time)
    sci_y_interp = interp1d(sol.t, sol.y[1], kind='cubic')(comparison_time)

    error_x = rk4_data[:len(comparison_time), 0] - sci_x_interp
    error_y = rk4_data[:len(comparison_time), 1] - sci_y_interp

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(rk4_data[:, 0], rk4_data[:, 1], "b-", label="Custom Fixed-Step RK4", alpha=0.6)
    ax1.plot(sol.y[0], sol.y[1], "r--", label="SciPy Adaptive RK45 Baseline", alpha=0.6)
    ax1.set_ylabel("Vertical Position Coordinate (m)")
    ax1.set_title("Cross-Validation Residual Architecture: Manual Integration vs Adaptive Baseline")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(comparison_time, error_x, "g-", label="Horizontal Coordinates Residuals")
    ax2.plot(comparison_time, error_y, "m-", label="Vertical Coordinates Residuals")
    ax2.set_xlabel("Temporal Phase Domain t (s)")
    ax2.set_ylabel("Absolute Coordinate Metric Residuals (m)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.savefig("plots/v2_1_scipy_validation.png", dpi=300)
    plt.show()
    print("Audit tracking logs finalized and recorded.")


if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")
    run_scipy_audit()
