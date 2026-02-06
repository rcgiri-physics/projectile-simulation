# Projectile motion: From Vacuum to Quadratic Drag

A computational physics study comparing analytical and numerical solution for motion in a fluid.

## Abstract

This project investigates the numerical simulation of projectile motion in dissipative media. We transition from the idealized vacuum model to non-linear drag regimes, implementing a 4th-order Runge-Kutta (RK4) integration scheme. By utilizing vectorized state-space representations in Python (NumPy), we demonstrate significant improvements in numerical stability and global truncation error ( $O(\Delta t^4)$ ) compared to first-order Euler methods. This repository serves as a foundational framework for modeling complex classical systems where analytical solutions are non-existent.

## Introduction

This project simulates the trajectory of a projectile under three different physical models. It demonstrates the transition from simple classroom physics to realistic, non-linear systems that require numerical integration.

**Note:** *Version 2.0 Update:* This project has been upgraded to a research-standard 4th-order Runge-Kutta (RK4) integrator. By sampling the derivatives at four different points per time-step, the global error has been reduced from $O(\Delta t)$ to $O(\Delta t^4)$, allowing for high-precision trajectories even with larger time steps.

*Version 1.0* implements a manual time-stepping algorithm (Euler Method) using standard Python data structures (lists) to avoid dependency on heavy numerical libraries. it is worth noting that the Euler method is a "first-order" numerical method. This means its global error is roughly proportional to the time step $\Delta t$. It is only accurate for very small time steps, so, later version we will move to *SciPy* and *RK4*.

## 1. Physical models

For all three cases, we start with Newton's Second Law:

$$\sum\vec{F}=m\vec{a} => m\frac{d\vec{v}}{dt}=m\vec{g}+\vec{F}_{drag}$$

To solve this numerically in Python, we split the second-order equation into two first-order equations for each dimension (x and y)

$$\frac{dx}{dt}=v_x, \frac{dy}{dt}=v_y$$

### Case A: Ideal motion (Vacuum)

In a vacuum, the only force is gravity

- Equation of motion

$$\frac{d\vec{v_x}}{dt} = 0,  \frac{d\vec{v_y}}{dt} = -\vec{g}$$

- Analytical Solution:

$$x(t) = v_{0x}t$$

$$y(t) = v_{0y}t - \frac{1}{2} gt^2$$

- Solving parameters: $g=9.81m/s^2$

- Physical Insight: The trajectory is a perfect parabola. The horizontal velocity $v_x$ remains constant throughout the flight.

### Case B: Linear Drag (Stokes' Drag)

Applicable for low-speed motion where $F_{drag} = -b\vec{v}$.

- Equation of motion

$$\frac{dv_x}{dt} = -kv_x, \quad \frac{dv_y}{dt} = -g - kv_y$$

- Analytical solution:

$$x(t) = \frac{v_{x0}}{k}(1 - e^{-kt})$$

- Solving Parameters:
$m$: Mass of the object.
$b = 6 \pi \eta r$ (where $\eta$ is the dynamic viscosity of the fluid).

- Physical Insight: The equations are uncoupled (the $x$-motion does not affect the $y$-motion). The object approaches a terminal velocity $v_t = \frac{mg}{b}$. It explains why we use different models. It notes that Linear Drag applies to tiny objects [cite](https://pubs.aip.org/aapt/ajp/article-abstract/67/6/538/1055298/On-the-rise-and-fall-of-a-ball-with-linear-or) (like dust or oil droplets, $Re < 1$) while Quadratic Drag applies to sports balls ($Re > 1000$). Fixed coefficients are used in the simulation for clarity only.

### Case C: Quadratic drag (Newtonian Drag)

The most realistic model for macroscopic objects where $\vec{F}_{drag} = -cv\vec{v}$.
$\vec{F}_{drag} = -\frac{1}{2} C_d \rho A |\vec{v}| \vec{v}$

- Differential Equations:
$$\frac{dv_x}{dt} = -\frac{1}{m} \left( \frac{1}{2} C_d \rho A \sqrt{v_x^2 + v_y^2} \right) v_x$$
$$\frac{dv_y}{dt} = -g - \frac{1}{m} \left( \frac{1}{2} C_d \rho A \sqrt{v_x^2 + v_y^2} \right) v_y$$

- Solving Parameters:
$\rho$: Density of air ($\approx 1.225 \, \text{kg/m}^3$ at sea level).
$C_d$: Drag coefficient ($0.47$ for a sphere).
$A$: Cross-sectional area ($\pi r^2$).

- Physical Insight: The equations are coupled because the total speed $|\vec{v}|$ affects both components. The object approaches a terminal velocity $v_t = \sqrt{\frac{2mg}{\rho C_d A}}$. There is no simple analytical solution; numerical integration (like RK4) is mandatory.

## 2. Numerical Implementation

Baseline: Euler Method (v1.0)

In classical mechanics, we describe motion using continuous differential equations: $$\frac{d\vec{v}}{dt} = \vec{a}$$

To solve this in Python, we discretize time into small steps $\Delta t$. We approximate the change in velocity and position as: $$v_{n+1} = v_n + a_n \Delta t$$
$$s_{n+1} = s_n + v_{n+1} \Delta t$$

Also, since the quadratic case has no closed-form analytical solution, we necessarily need to use numerical (version 1.0) methods to solve this problem.

In **Version 2.0**, the simulation utilizes State-Vector notation. Instead of updating $x$ and $y$ as separate variables, we define a state vector $\mathbf{s}$:

$$\mathbf{s} = [x, y, v_x, v_y]$$

The time evolution is governed by the 4th-order Runge-Kutta algorithm:

$$\mathbf{k_1} = \mathbf{f}(t_n, \mathbf{s}_n)$$

$$\mathbf{k_2} = \mathbf{f}(t_n + \frac{\Delta t}{2}, \mathbf{s}_n + \frac{\Delta t}{2}\mathbf{k_1})$$

$$\mathbf{k_3} = \mathbf{f}(t_n + \frac{\Delta t}{2}, \mathbf{s}_n + \frac{\Delta t}{2}\mathbf{k_2})$$

$$\mathbf{k_4} = \mathbf{f}(t_n + \Delta t, \mathbf{s}_n + \Delta t\mathbf{k_3})$$

$$\mathbf{s}_{n+1} = \mathbf{s}_n + \frac{\Delta t}{6}(\mathbf{k_1} + 2\mathbf{k_2} + 2\mathbf{k_3} + \mathbf{k_4})$$

By implementing this using NumPy, we achieve higher computational efficiency through vectorized operations, mirroring the mathematical notation used in classical mechanics research.

## Validation

To ensure the solver's accuracy, I compared the **Linear Drag** numerical output against the exact exponential analytical solution. The error was minimized to 10⁻⁶, confirming the code's reliability.

## 3. Results and Visualization

The simulation generates three key insights:

- Reliability (Validation)

We validated the Euler solver against the exact analytical solution for Linear Drag. As shown below, the numerical error is negligible for our time step.
![Validation Plot](plots/1_validation_error.png)

- Physical Models Comparison

Comparing the three drag models shows that quadratic drag (realistic air resistance) significantly reduces range compared to the vacuum ideal.
Version 1: ![Comparison Plot](plots/2_model_comparison.png)

Version 2: ![Comparison Plot](plots/v2_models.png)

- Effect of Mass

Unlike in a vacuum, mass matters in air. Heavier objects (blue/green) maintain momentum better and travel further than lighter objects (red/orange) which are easily slowed by air resistance.
![Mass Plot](plots/3_mass_variation.png)

- Numerical Stability Test

The stability check demonstrates that while the Euler method (first-order) begins to drift at $dt=0.1s$, the RK4 method maintains the physical integrity of the trajectory.
![Stability Check](plots/v2_stability_check.png)
Fig 4: A comparison of the Euler vs. RK4 method at a coarse time-step ($dt=0.1$). Note how the Euler method (dashed) over-calculates the trajectory, while the RK4 method maintains the physical energy of the system.

## Mathematical Appendix: Numerical Error Scaling

In this version, I transitioned from the Euler Method to the 4th-order Runge-Kutta (RK4) method. The primary motivation is the reduction of the Local Truncation Error (LTE).

1. The Euler Method (First-Order)The Euler method is a first-order Taylor expansion. It assumes the slope is constant over the entire interval $\Delta t$:$$y_{n+1} = y_n + f(t_n, y_n)\Delta t + O(\Delta t^2)$$

The Local Truncation Error is $O(\Delta t^2)$, which means if you halve the time-step, the error per step decreases by 4, but the number of steps doubles. Therefore, the Global Error is only $O(\Delta t)$.2. The Runge-Kutta Method (Fourth-Order)RK4 samples the derivative (acceleration) at four different points within each time-step to cancel out lower-order error terms:

$$k_1 = f(t_n, y_n)$$

$$k_2 = f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_1)$$

$$k_3 = f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_2)$$$$k_4 = f(t_n + \Delta t, y_n + \Delta t k_3)$$

$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

The Local Truncation Error for RK4 is $O(\Delta t^5)$, and the Global Error is $O(\Delta t^4)$.

*Comparison:* If we reduce the time-step by a factor of 10:

Euler's error decreases by 10 times.

RK4's error decreases by 10,000 times.

## 4. How to Run

1. Ensure you have Python installed.
2. Install dependencies: `pip install matplotlib numpy`
3. Run the script: `python projectile_sim_v2.py`

## 5. Future Enhancements

1. Using Python libraries that solve ODEs like `scipy.integrate.solve_ivp` and validate {Version 2.1}
2. Incorporate the Magnus Effect (spin on the ball)
3. Add a Variable Density Atmosphere model for high-altitude projectiles
4. Implement a GUI using TKinter or PyQt for real-time parameter adjustment

## 6. References & Resources

- For the Core Physics: Taylor, J.R. (2005). *Classical Mechanics.* University Science Books. (Chapter 2 covers Projectiles and Air Resistance.)
- For Fluid Dynamics (Drag): [NASA Glenn Research Center. The Drag Equation.](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-of-a-sphere/)
- For Numerical Methods: Press, W.H., et al. *Numerical Recipes: The Art of Scientific Computing.* Cambridge University Press

## 7. Conclusion: Insight from the simulation

The primary objective of this project was to transition from the idealized "vacuum" model of projectile motion to a realistic simulation incorporating fluid dynamics. Several key physical insights were discovered through the numerical results:

1. The Breakdown of Symmetry
In a vacuum, the trajectory is a perfectly symmetric parabola. However, once air resistance (Linear or Quadratic) is introduced, this symmetry is broken. The "descent" phase is always steeper than the "ascent" phase because the drag force continuously drains the kinetic energy of the system, leading to a shorter range and a lower peak height.
2. Optimal Launch Angle
While the theoretical maximum range in a vacuum is achieved at 45 degrees, this simulation demonstrates that in a medium with quadratic drag, the optimal angle shifts downward. For a standard projectile (like a ball), the maximum range is typically found between 35 and 40 degrees. This is because a lower angle reduces the "Time of Flight", thereby reducing the total time the drag force has to act on the horizontal velocity.
3. Numerical Reliability
By comparing the numerical results of the Linear Drag model with its Analytical Solution, the error was found to be in the order of 10⁻⁶ or less.
4. Mass Variation
Heavier objects have more inertia relative to their surface area, making them less affected by air resistance than lighter objects.
5. Final Reflection
This project demonstrates the necessity of computational methods in Physics. While simple models can be solved with pen and paper, "real-world" physics—where the forces are coupled and non-linear—requires the power of numerical integration.

This simulation serves as a foundational step toward more complex models involving the Magnus Effect (lift from spin) or Variable Air Density.
