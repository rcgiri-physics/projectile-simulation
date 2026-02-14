import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

# 1. ATTEMPT THE IMPORT
try:
    from projectile_sim_v2 import ProjectileV2
    print("✓ Physics Engine loaded successfully.")
except ImportError:
    print("X ERROR: Could not find 'projectile_sim_v2.py'.")
    exit()

def ground_event(t, state):
    """Detects ground impact (y <= 0)."""
    return state[1]

# This is the critical fix:
# terminal = True stops the simulation.
# direction = -1 tells SciPy to ONLY trigger when y is DECREASING (falling).
# This prevents it from stopping at the very start (t=0, y=0).
ground_event.terminal = True
ground_event.direction = -1 

def run_scipy_audit():
    sim = ProjectileV2(mass=0.45, radius=0.11)
    v0, angle = 30, 45
    dt = 0.01
    
    print("Starting Version 2.1 Audit...")

    # 1. Custom RK4 Data
    rk4_data = sim.simulate(v0, angle, mode='quadratic', dt=dt, method='rk4')
    t_rk4 = np.arange(len(rk4_data)) * dt
    
    # 2. SciPy Benchmark
    rad = np.radians(angle)
    # Start slightly above 0 (1mm) to ensure we don't trigger the ground event at t=0
    initial_state = [0.0, 0.001, v0 * np.cos(rad), v0 * np.sin(rad)]
    
    def system_dynamics(t, state):
        v_vec = state[2:]
        accel = sim._get_accel(v_vec, mode='quadratic')
        return [state[2], state[3], accel[0], accel[1]]
    
    sol = solve_ivp(
        system_dynamics, 
        t_span=(0, 20), 
        y0=initial_state, 
        method='RK45', 
        events=ground_event, 
        rtol=1e-10, 
        atol=1e-12
    )
    
    # 3. Validation Check
    if sol.t.size < 2:
        print("X ERROR: SciPy solver stopped at start. Adjusting height...")
        return

    # 4. Interpolation Logic
    max_common_t = min(t_rk4[-1], sol.t[-1])
    comparison_time = np.arange(0, max_common_t, dt)
    
    sci_x_interp = interp1d(sol.t, sol.y[0])(comparison_time)
    sci_y_interp = interp1d(sol.t, sol.y[1])(comparison_time)
    
    # 5. Calculate Residuals
    error_x = rk4_data[:len(comparison_time), 0] - sci_x_interp
    error_y = rk4_data[:len(comparison_time), 1] - sci_y_interp

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Visual Comparison
    ax1.plot(rk4_data[:,0], rk4_data[:,1], 'b-', label='Custom RK4 (v2.0)', alpha=0.6)
    ax1.plot(sol.y[0], sol.y[1], 'r--', label='SciPy solve_ivp', alpha=0.6)
    ax1.set_ylabel("Height (m)")
    ax1.set_title("Trajectory Audit: Custom RK4 vs. SciPy RK45")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Precision Audit (Residuals)
    ax2.plot(comparison_time, error_x, label='X-Error', color='forestgreen')
    ax2.plot(comparison_time, error_y, label='Y-Error', color='darkmagenta')
    ax2.set_ylabel("Absolute Error (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Residual Analysis (Max Error: {np.max(np.abs(error_y)):.2e} m)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if not os.path.exists('plots'): os.makedirs('plots')
    plt.savefig('plots/v2_1_scipy_validation.png')
    print(f"Audit complete! Max error: {np.max(np.abs(error_y)):.2e} meters.")

if __name__ == "__main__":
    run_scipy_audit()