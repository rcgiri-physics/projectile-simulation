import math
import matplotlib.pyplot as plt

# ==========================================
# 1. The Physics Class (Simplified)
# ==========================================
class Projectile:
    def __init__(self, mass, radius, drag_coeff=0.47, viscosity=1.81e-5):
        self.mass = mass
        self.radius = radius
        self.g = 9.81
        self.rho = 1.225
        
        # --- CASE B: Linear Drag (Stokes) ---
        # b = 6 * pi * eta * r
        #self.b_stokes = 6 * math.pi * viscosity * radius
        self.b_stokes = 0.01  # For testing purposes, override with a fixed value
        self.k_linear = self.b_stokes / self.mass

        # --- CASE C: Quadratic Drag (Newton) ---
        # k = 0.5 * rho * Cd * Area / mass
        area = math.pi * (radius ** 2)
        self.k_quad = (0.5 * self.rho * drag_coeff * area) / self.mass

    def simulate(self, v0, angle, mode='vacuum', dt=0.001):
        # Convert angle to radians
        rad = math.radians(angle)
        vx = v0 * math.cos(rad)
        vy = v0 * math.sin(rad)
        
        x, y, t = 0.0, 0.0, 0.0
        
        # Simple lists to store history
        t_list = [t]
        x_list = [x]
        y_list = [y]

        while y >= 0:
            # 1. Calculate Acceleration
            if mode == 'vacuum':
                ax, ay = 0, -self.g
            elif mode == 'linear':
                ax = -self.k_linear * vx
                ay = -self.g - (self.k_linear * vy)
            elif mode == 'quadratic':
                speed = math.sqrt(vx**2 + vy**2)
                ax = -self.k_quad * speed * vx
                ay = -self.g - (self.k_quad * speed * vy)
            
            # 2. Update Position & Velocity (Euler Step)
            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt
            t += dt
            
            # Save if above ground
            if y >= 0:
                t_list.append(t)
                x_list.append(x)
                y_list.append(y)
        
        return t_list, x_list, y_list

# ==========================================
# 2. Plotting Functions
# ==========================================

def plot_validation_error(ball, v0, angle):
    """
    PLOT 1: Validates Linear Drag against Exact Math and shows Error.
    """
    print("Generating Plot 1: Validation & Error...")
    
    # Run Numerical Simulation
    t_list, x_list, y_list = ball.simulate(v0, angle, mode='linear')
    
    # Calculate Exact Math Solution for comparison
    vy0 = v0 * math.sin(math.radians(angle))
    k = ball.k_linear
    g = ball.g
    
    y_exact = []
    errors = []
    
    # Loop through time steps to calculate exact value and error
    for i in range(len(t_list)):
        t = t_list[i]
        
        # Exact Formula for Linear Drag Height
        term1 = (1/k) * (vy0 + (g/k)) * (1 - math.exp(-k * t))
        term2 = (g * t) / k
        val = term1 - term2
        
        y_exact.append(val)
        errors.append(abs(y_list[i] - val))

    # Create the Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top Subplot: Trajectories
    ax1.plot(t_list, y_list, 'bo', markersize=2, label='Euler Method')
    ax1.plot(t_list, y_exact, 'r-', linewidth=1, label='Exact Analytical')
    ax1.set_title("Validation: Linear Drag Model")
    ax1.set_ylabel("Height (m)")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom Subplot: Error
    ax2.plot(t_list, errors, color='purple')
    ax2.set_title("Absolute Error (Numerical vs Analytical)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('projectile-simulation/plots/1_validation_error.png', dpi=300)
    plt.show()

def plot_model_comparison(ball, v0, angle):
    """
    PLOT 2: Compares Vacuum vs Linear vs Quadratic.
    """
    print("Generating Plot 2: Model Comparison...")
    
    # Run all 3 modes
    _, x_vac, y_vac = ball.simulate(v0, angle, 'vacuum')
    _, x_lin, y_lin = ball.simulate(v0, angle, 'linear')
    _, x_quad, y_quad = ball.simulate(v0, angle, 'quadratic')
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vac, y_vac, 'k--', label='Vacuum (Ideal)')
    plt.plot(x_lin, y_lin, 'g-.', label='Linear Drag')
    plt.plot(x_quad, y_quad, 'b-', linewidth=2, label='Quadratic Drag')
    
    plt.title("Trajectory Comparison: Vacuum vs Air Resistance")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.axhline(0, color='black')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.fill_between(x_quad, y_quad, alpha=0.1, color='blue')
    
    plt.savefig('projectile-simulation/plots/2_model_comparison.png', dpi=300)
    plt.show()

def plot_mass_variation(v0):
    """
    PLOT 3: Shows how Mass and Angle affect the trajectory.
    """
    print("Generating Plot 3: Mass & Angle Variation...")
    
    plt.figure(figsize=(10, 6))
    
    # Define scenarios: [Mass, Angle, Color, Label]
    scenarios = [
        [0.05, 45, 'red',   'Light (0.05kg) - 45°'],
        [5.00, 45, 'blue',  'Heavy (5.0kg)  - 45°'],
        [0.05, 30, 'orange','Light (0.05kg) - 30°'],
        [5.00, 30, 'green', 'Heavy (5.0kg)  - 30°']
    ]
    
    for mass, angle, color, label in scenarios:
        # Create a new ball for each scenario
        temp_ball = Projectile(mass=mass, radius=0.05)
        
        # Simulate (using Quadratic because it's the most realistic)
        _, x, y = temp_ball.simulate(v0, angle, mode='quadratic')
        
        plt.plot(x, y, color=color, label=label)

    plt.title(f"Effect of Mass on Trajectory (Quadratic Drag, v0={v0}m/s)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.axhline(0, color='black')
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.savefig('projectile-simulation/plots/3_mass_variation.png', dpi=300)
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    import os
    # Create the directory if it doesn't exist
    os.makedirs('projectile-simulation/plots', exist_ok=True)
    # Settings
    v0 = 50
    angle = 45
    
    # Create a standard ball (0.15 kg)
    my_ball = Projectile(mass=0.15, radius=0.037)

    # 1. Run Validation Plot
    plot_validation_error(my_ball, v0, angle)
    
    # 2. Run Comparison Plot
    plot_model_comparison(my_ball, v0, angle)
    
    # 3. Run Mass Variation Plot
    plot_mass_variation(v0)