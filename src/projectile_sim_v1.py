"""
Kinematic Simulation Engine (First-Order Baseline).

This module simulates two-dimensional projectile trajectories across vacuum,
linear (Stokes), and quadratic (Newtonian) aerodynamic resistance regimes using
a standard first-order forward Euler time-stepping routine. It relies entirely
on built-in Python data structures to establish a baseline verification framework
free of external array dependencies.
"""

import math
import matplotlib.pyplot as plt


class Projectile:
    """
    Represent a spherical macroscopic projectile moving through fluid media.

    Calculates damping coefficients derived from physical dimensions and fluid
    properties, managing the discrete integration of uncoupled or linearly
    coupled equations of motion.
    """

    def __init__(self, mass, radius, drag_coefficient=0.47, viscosity=1.81e-5):
        """
        Initialize system parameters and pre-calculate drag scaling limits.

        Parameters:
        -----------
        mass : float
            Inertial mass of the projectile in kilograms (kg).
        radius : float
            Spherical radius of the projectile in meters (m).
        drag_coefficient : float, optional
            Dimensionless aerodynamic drag coefficient (sphere baseline = 0.47).
        viscosity : float, optional
            Dynamic fluid viscosity of the medium in Pascal-seconds (Pa*s).
            Defaults to sea-level ambient air value (1.81e-5 Pa*s).
        """
        self.mass = mass
        self.radius = radius
        self.g = 9.81
        self.rho = 1.225

        self.b_stokes = 6 * math.pi * viscosity * radius
        self.k_linear = self.b_stokes / self.mass

        area = math.pi * radius**2
        self.k_quadratic = (0.5 * self.rho * area * drag_coefficient) / self.mass

    def simulate(self, v0, angle, mode="vacuum", dt=0.001):
        """
        Execute discrete first-order forward Euler numerical step integration.

        Iterates kinematic parameters forward in time until vertical position
        crosses the lower boundary plane (y < 0).

        Parameters:
        -----------
        v0 : float
            Initial launch velocity magnitude in meters per second (m/s).
        angle : float
            Launch angle relative to the horizontal plane in degrees.
        mode : str, optional
            Environmental regime selection: 'vacuum', 'linear', or 'quadratic'.
        dt : float, optional
            Fixed temporal step size increment in seconds (s).

        Returns:
        --------
        t_list : list of float
            Discretized temporal steps from launch to impact boundary.
        x_list : list of float
            Horizontal displacement coordinates (m) mapped over time.
        y_list : list of float
            Vertical displacement coordinates (m) mapped over time.
        """
        rad = math.radians(angle)
        vx = v0 * math.cos(rad)
        vy = v0 * math.sin(rad)

        x, y, t = 0.0, 0.0, 0.0

        x_list = [x]
        y_list = [y]
        t_list = [t]

        while y >= 0:
            v = math.sqrt(vx**2 + vy**2)

            if mode == "vacuum":
                ax = 0
                ay = -self.g
            elif mode == "linear":
                ax = -self.k_linear * vx
                ay = -self.g - self.k_linear * vy
            elif mode == "quadratic":
                ax = -self.k_quadratic * v * vx
                ay = -self.g - self.k_quadratic * v * vy
            else:
                raise ValueError(f"Unsupported mode '{mode}' specified. Choose 'vacuum', 'linear', or 'quadratic'.") #pylint: disable=line-too-long

            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt
            t += dt

            if y >= 0:
                x_list.append(x)
                y_list.append(y)
                t_list.append(t)

        return t_list, x_list, y_list


def run_simulations():
    """
    Generate comparative multi-regime kinematic profiles.

    Simulates trajectories for a standard test projectile across vacuum,
    linear, and quadratic regimes, exporting the comparative visualization
    to the workspace directory.
    """
    ball = Projectile(mass=0.145, radius=0.037)
    v0 = 30.0
    angle = 45.0

    _, x_vac, y_vac = ball.simulate(v0, angle, mode="vacuum")
    _, x_lin, y_lin = ball.simulate(v0, angle, mode="linear")
    _, x_qdr, y_qdr = ball.simulate(v0, angle, mode="quadratic")

    plt.figure(figsize=(10, 5))
    plt.plot(x_vac, y_vac, "k--", label="Vacuum (Analytical Baseline)")
    plt.plot(x_lin, y_lin, "b-", label="Linear Drag (Stokes)")
    plt.plot(x_qdr, y_qdr, "r-", label="Quadratic Drag (Newton)")

    plt.title("Kinematic Profiles Across Dissipative Regimes")
    plt.xlabel("Horizontal Displacement x (m)")
    plt.ylabel("Vertical Displacement y (m)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("plots/1_model_comparison.png", dpi=300)
    plt.show()


def run_angle_sensitivity():
    """
    Evaluate optimal launch angle variance under quadratic aerodynamic resistance.

    Sweeps through multiple angular initialization states to visually map
    the breakdown of classic symmetric horizontal displacement properties.
    """
    ball = Projectile(mass=0.145, radius=0.037)
    v0 = 30.0
    angles = [30, 35, 40, 45, 50]

    plt.figure(figsize=(10, 5))
    for alpha in angles:
        _, x, y = ball.simulate(v0, alpha, mode="quadratic")
        plt.plot(x, y, label=f"Launch Angle = {alpha}°")

    plt.title("Launch Angle Sensitivity Optimization Under Quadratic Drag")
    plt.xlabel("Horizontal Displacement x (m)")
    plt.ylabel("Vertical Displacement y (m)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("plots/2_angle_sensitivity.png", dpi=300)
    plt.show()


def run_mass_sensitivity():
    """
    Isolate mass-dependent inertial scaling properties within fluid vector fields.

    Simulates low-mass and high-mass projectile responses under identical
    aerodynamic cross-sections to map geometric deceleration behavior.
    """
    v0 = 30.0
    scenarios = [
        [0.05, 45, "red", "m = 0.05kg, alpha = 45°"],
        [5.00, 45, "blue", "m = 5.00kg, alpha = 45°"],
        [0.05, 30, "orange", "m = 0.05kg, alpha = 30°"],
        [5.00, 30, "green", "m = 5.00kg, alpha = 30°"],
    ]

    plt.figure(figsize=(10, 5))
    for mass, angle, color, label in scenarios:
        temp_ball = Projectile(mass=mass, radius=0.05)
        _, x, y = temp_ball.simulate(v0, angle, mode="quadratic")
        plt.plot(x, y, color=color, label=label)

    plt.title("Mass Variation and Inertial Deviations in Quadratic Regimes")
    plt.xlabel("Horizontal Displacement x (m)")
    plt.ylabel("Vertical Displacement y (m)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("plots/3_mass_variation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    import os

    if not os.path.exists("plots"):
        os.makedirs("plots")
    run_simulations()
    run_angle_sensitivity()
    run_mass_sensitivity()
