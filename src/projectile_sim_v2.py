"""
Vectorized Vector State-Space Integration Framework (High-Precision Engine).

This module details a refactored kinematic processing framework utilizing
NumPy array operations and a fourth-order Runge-Kutta (RK4) time-integration
routine. Kinematics are tracked across an explicit state-space matrix layer,
bounding local truncation accumulation to O(dt^4).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class ProjectileV2:
    """
    High-precision vectorized projectile flight simulation engine.

    Manages system multi-variable equations of motion within state vector
    formulations, performing explicit fixed-step numerical integration loops.
    """

    def __init__(self, mass, radius, C_d=0.47):  # pylint: disable=invalid-name
        """
        Initialize material and aerodynamic coordinates for state-space scaling.

        Parameters:
        -----------
        mass : float
            Projectile mass metric evaluated in kilograms (kg).
        radius : float
            Projectile radius dimension recorded in meters (m).
        C_d : float, optional
            Dimensionless drag profile coefficient (defaults to spherical 0.47).
        """
        self.m = mass
        self.r = radius
        self.g = 9.81
        self.rho = 1.225
        self.area = np.pi * (radius**2)

        self.k_lin = 0.01 / self.m
        self.k_quad = (0.5 * self.rho * C_d * self.area) / self.m

    def _state_derivative(self, z, mode):
        """
        Compute continuous first-order differential system state derivative vector.

        Decomposes velocities and acceleration coordinates according to selected
        drag regimes.

        Parameters:
        -----------
        z : ndarray, shape (4,)
            Current system state vector mapped as [x, y, vx, vy]^T.
        mode : str
            Target atmospheric model definition: 'vacuum', 'linear', 'quadratic'.

        Returns:
        --------
        dzdt : ndarray, shape (4,)
            Time-derivative vector profile evaluated as [vx, vy, ax, ay]^T.
        """
        v_components = z[2:]
        v_mag = np.linalg.norm(v_components)

        dzdt = np.zeros(4)
        dzdt[0:2] = v_components
        dzdt[2] = 0.0
        dzdt[3] = -self.g

        if mode == "linear":
            dzdt[2:] -= self.k_lin * v_components
        elif mode == "quadratic":
            dzdt[2:] -= self.k_quad * v_mag * v_components

        return dzdt

    def simulate(self, initial_velocity, angle, mode="quadratic", dt=0.01, method="rk4"):  # pylint: disable=redefined-outer-name
        """
        Track trajectory boundaries across continuous state fields.

        Supports comparative stepping routines using either first-order Euler
        or fourth-order Runge-Kutta configurations.

        Parameters:
        -----------
        initial_velocity : float
            Initial trajectory speed magnitude in meters per second (m/s).
        angle : float
            Inclination launch component measured in degrees.
        mode : str, optional
            Drag regime specification selection ('vacuum', 'linear', 'quadratic').
        dt : float, optional
            Discrete integration step constraint interval in seconds (s).
        method : str, optional
            Target numerical processor engine framework selection ('rk4', 'euler').

        Returns:
        --------
        trajectory : ndarray, shape (N, 4)
            Compiled state array documenting localized position and velocity
            vectors over time.
        """
        rad = np.radians(angle)
        z = np.array([0.0, 0.0, initial_velocity * np.cos(rad), initial_velocity * np.sin(rad)])

        trajectory = [z.copy()]

        while z[1] >= 0:
            if method == "euler":
                dzdt = self._state_derivative(z, mode)
                z += dzdt * dt
            elif method == "rk4":
                k1 = self._state_derivative(z, mode)
                k2 = self._state_derivative(z + 0.5 * dt * k1, mode)
                k3 = self._state_derivative(z + 0.5 * dt * k2, mode)
                k4 = self._state_derivative(z + dt * k3, mode)
                z += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                raise ValueError(f"Unsupported integration method: {method}")

            if z[1] >= 0:
                trajectory.append(z.copy())
            else:
                fraction = -z[1] / (trajectory[-1][3] if len(trajectory) > 0 else 1.0)
                if int(fraction) != 0:
                    z_final = z + fraction * self._state_derivative(z, mode) * dt
                    trajectory.append(z_final)
                break

        return np.array(trajectory)


if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")

    sim = ProjectileV2(mass=0.45, radius=0.11)
    v0, angle = 30.0, 45.0

    plt.figure(figsize=(10, 5))
    for m in ["vacuum", "linear", "quadratic"]:
        data = sim.simulate(v0, angle, mode=m, method="rk4")
        plt.plot(data[:, 0], data[:, 1], label=f"RK4 Engine: {m.capitalize()}")

    plt.title("Kinematic Discretization Phase Path Comparison")
    plt.xlabel("Horizontal Coordinate q_x (m)")
    plt.ylabel("Vertical Coordinate q_y (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/v2_models.png")

    plt.figure(figsize=(10, 5))
    rough_dt = 0.1  # pylint: disable=invalid-name

    euler_data = sim.simulate(v0, angle, dt=rough_dt, method="euler")
    rk4_data = sim.simulate(v0, angle, dt=rough_dt, method="rk4")

    plt.plot(euler_data[:, 0], euler_data[:, 1], "r--", label=r"First-Order Euler ($\Delta t = 0.1\,$s)") #pylint: disable=line-too-long
    plt.plot(rk4_data[:, 0], rk4_data[:, 1], "b-", label=r"Fourth-Order RK4 ($\Delta t = 0.1\,$s)")

    plt.title("Algorithmic Divergence and Integration Stability Constraints")
    plt.xlabel("Horizontal Coordinate q_x (m)")
    plt.ylabel("Vertical Coordinate q_y (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/v2_stability_check.png")
    plt.show()
