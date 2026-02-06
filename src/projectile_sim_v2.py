import numpy as np
import matplotlib.pyplot as plt
import os

class ProjectileV2:
    """
    Refactored Projectile class using NumPy for vectorization 
    and RK4 for numerical stability.
    """
    def __init__(self, mass, radius, C_d=0.47):
        self.m = mass
        self.r = radius
        self.g = 9.81
        self.rho = 1.225
        self.area = np.pi * (radius**2)
        
        # Pre-calculating constants to save clock cycles in the loop
        self.k_lin = 0.01 / self.m  # Linear drag constant
        self.k_quad = (0.5 * self.rho * C_d * self.area) / self.m

    def _get_accel(self, v_vec, mode):
        """Internal helper to calculate acceleration vector."""
        v_mag = np.linalg.norm(v_vec)
        accel = np.array([0, -self.g]) # Gravity is always there
        
        if mode == 'linear':
            return accel - self.k_lin * v_vec
        elif mode == 'quadratic':
            return accel - self.k_quad * v_mag * v_vec
        return accel # Vacuum case

    def simulate(self, v0, angle, mode='quadratic', dt=0.01, method='rk4'):
        """
        Main simulation loop. Supports both 'euler' and 'rk4'.
        """
        rad = np.radians(angle)
        # s = [x, y, vx, vy]
        s = np.array([0.0, 0.0, v0 * np.cos(rad), v0 * np.sin(rad)])
        
        history = [s.copy()]
        
        while s[1] >= 0:
            if method == 'euler':
                # --- The Euler Step (Old Reliable) ---
                v_current = s[2:]
                a_current = self._get_accel(v_current, mode)
                
                s[0:2] += v_current * dt # Update pos
                s[2:4] += a_current * dt # Update vel
                
            else:
                # --- The RK4 Step (The Precision Move) ---
                # k values represent [delta_pos, delta_vel]
                def derivatives(state):
                    v = state[2:]
                    a = self._get_accel(v, mode)
                    return np.array([v[0], v[1], a[0], a[1]])

                k1 = derivatives(s)
                k2 = derivatives(s + (dt/2) * k1)
                k3 = derivatives(s + (dt/2) * k2)
                k4 = derivatives(s + dt * k3)
                
                s += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

            history.append(s.copy())
            
        return np.array(history)

# ==========================================================
# RESEARCH & TESTING SCRIPT
# ==========================================================
if __name__ == "__main__":
    # Ensure local directory for output exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Setup a standard test: A soccer ball sized projectile
    sim = ProjectileV2(mass=0.45, radius=0.11) 
    v0, angle = 30, 45
    
    # 1. Compare Physics Models
    plt.figure(figsize=(10, 5))
    for m in ['vacuum', 'linear', 'quadratic']:
        data = sim.simulate(v0, angle, mode=m, method='rk4')
        plt.plot(data[:,0], data[:,1], label=f'RK4: {m.capitalize()}')
    
    plt.title("Physics Model Comparison (V2.0 RK4)")
    plt.xlabel("X Displacement (m)")
    plt.ylabel("Y Displacement (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('projectile-simulation/plots/v2_models.png')
    
    # 2. Stability Check: Euler vs RK4 at a rough time-step
    plt.figure(figsize=(10, 5))
    rough_dt = 0.1 # This is quite large, good for showing error
    
    euler_data = sim.simulate(v0, angle, dt=rough_dt, method='euler')
    rk4_data = sim.simulate(v0, angle, dt=rough_dt, method='rk4')
    
    plt.plot(euler_data[:,0], euler_data[:,1], 'r--', label='Euler (Rough)')
    plt.plot(rk4_data[:,0], rk4_data[:,1], 'b-', label='RK4 (Precise)')
    
    plt.title(f"Numerical Stability Test (dt = {rough_dt}s)")
    plt.legend()
    plt.savefig('projectile-simulation/plots/v2_stability_check.png')
    
    print("Simulations complete. Check /plots/ for results.")