import numpy as np
from typing import Callable
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

class HomeRun:
    def __init__(
        self, 
        m: float, 
        g: float, 
        rho: float, 
        c_d: float, 
        c_l: float, 
        r: float,
        h: float,
        s: float,
        theta: float, 
        phi: float,
        omega: np.ndarray = np.array([0, -2000, 0])
    ):
        """
        class attributes:

            m: Mass of baseball
            g: Acceleration due to gravity
            rho: Air density
            c_d: Drag coefficient
            c_l: Lift coefficient
            r: Radius of baseball
            h: Height a baseball is hit from above home plate
            s: Initial speed of the baseball
            theta: Upward (vertical) angle of the batted ball, in degrees
            phi: Heading (horizontal) angle of the batted ball, in degrees
            omega: Spin vector in RPM (defaulting to some backspin)
            R: A dictionary that maps heading angle (phi) values to the appropriate distance from home plate to the outfield wall
            H: A dictionary that maps heading angle (phi) values to the appropriate outfield wall height
        """
        self.m = m
        self.g = g
        self.rho = rho
        self.c_d = c_d
        self.c_l = c_l
        self.r = r
        self.h = h
        self.s = s
        self.theta = theta
        self.phi = phi 
        self.omega = omega * (2 * np.pi / 60)

        self.t_arr = None
        self.traj_arr = None

        self._t_max = None
        self._interp_x = None
        self._interp_y = None
        self._interp_z = None

        self.R = None
        self.H = None

        self._validate_initial_params()

    def _validate_initial_params(self):
        """Internal helper method to validate initial class parameters"""
        if self.m <= 0:
            raise ValueError("Mass (m) must be positive.")
        if not (0 <= self.phi <= 180):
            raise ValueError(f"Phi ({self.phi}) must be between 0 and 180 degrees.")
        if self.s < 0:
            raise ValueError("Initial speed (s) cannot be negative.")
        if self.g <= 0:
            raise ValueError("Acceleration due to gravity (g) must be positive.")
        if self.rho < 0:
            raise ValueError("Air density (rho) cannot be negative.")
        if self.r <= 0:
            raise ValueError("Radius (r) must be positive.")
        if self.h < 0:
            raise ValueError("Initial height (h) cannot be negative.")
        if not (-90 <= self.theta <= 90):
            raise ValueError(f"Theta ({self.theta}) must be between -90 and 90 degrees.")
        if not isinstance(self.omega, np.ndarray) or self.omega.shape != (3,):
            raise TypeError("Spin vector (omega) must be a numpy array of shape (3,).")
        
    def _ode(
        self,
        _t: float,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Ordinary First Order System of Differential Equations used to model the 
        flight path of a batted baseball

        Args:
            time: current time
            y: current state vector
        """
        v_vec = y[3:6]
        v_mag = np.linalg.norm(v_vec)

        f = np.zeros(6)

        kd = self.c_d * self.rho * np.pi * self.r**2 / (self.m * 2)
        kl = self.c_l * self.rho * np.pi * self.r**2 / (self.m * 2)

        omega_cross_v = np.cross(self.omega, v_vec)
        omega_mag = np.linalg.norm(self.omega)
        if omega_mag > 0:
            lift_dir = omega_cross_v / (omega_mag * v_mag + 1e-12)
        else:
            lift_dir = np.zeros(3)  

        drag_accel = -kd * v_mag * v_vec
        lift_accel = kl * v_mag * omega_mag * lift_dir

        f[0] = v_vec[0]
        f[1] = v_vec[1]
        f[2] = v_vec[2]
        f[3] = drag_accel[0] + lift_accel[0]
        f[4] = drag_accel[1] + lift_accel[1]
        f[5] = drag_accel[2] + lift_accel[2] - self.g

        return f

    def _rk4_step(
        self, 
        f: Callable[[float, np.ndarray], np.ndarray],
        dt: float, 
        t: float,
        y: np.ndarray, 
    ) -> np.ndarray:
        """
        Fourth-Order Runge-Kutta Method for solving ODEs

        Args:
            f: the ODE function, e.g., f(t, y)
            dt: time step
            t: current time
            y: current state vector
        """
        
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)

        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    def solve_trajectory(
        self,
        dt: float,
        maxiter: int = 10000,
    ) -> np.ndarray:
        """
        Solves the system of ODEs using an RK4 method, and 
        sets cubic spline interpolations for x(t), y(t), z(t) functions

        Args:
            dt: time step
            maxiter: maximum number of iterations for RK4
        """
        
        theta_rad = np.radians(self.theta)
        phi_rad = np.radians(self.phi)
        
        vx0 = self.s * np.cos(theta_rad) * np.cos(phi_rad)
        vy0 = self.s * np.cos(theta_rad) * np.sin(phi_rad)
        vz0 = self.s * np.sin(theta_rad)
        
        # Initial conditions
        y = np.array([0.0, 0.0, self.h, vx0, vy0, vz0])
        
        t_history = [0.0]
        trajectory = [y.copy()]
        
        t_elapsed = 0.0
        
        for _ in range(maxiter):
            y = self._rk4_step(self._ode, dt, t_elapsed, y)
            t_elapsed += dt
            
            t_history.append(t_elapsed)
            trajectory.append(y.copy())
            
            # Physical exit condition
            if y[2] <= 0.0:
                break
                
        self.t_arr = np.array(t_history)
        self.traj_arr = np.array(trajectory)
        self._t_max = t_elapsed
        
        # Cubic spline interpolations for x(t), y(t), z(t) functions
        self._interp_x = CubicSpline(self.t_arr, self.traj_arr[:, 0])
        self._interp_y = CubicSpline(self.t_arr, self.traj_arr[:, 1])
        self._interp_z = CubicSpline(self.t_arr, self.traj_arr[:, 2])
            
    def set_R(
        self, 
        angle_ls: list[float], 
        dist_ls: list[float]
    ) -> None:
        """
        Sets the R dictionary, where keys are sectors represented by tuples of angles and
        values are the distance to the outfield wall for that sector
        
        Args:
            angle_ls: list of angles, expects ordered
            dist_ls: list of distances, expects ordered
        """
        if not isinstance(angle_ls, list) or len(angle_ls) < 2:
            raise ValueError("'angle_ls' must be list type with two or more values representing heading angles 'phi'.")
        
        if not isinstance(dist_ls, list) or len(dist_ls) < 1:
            raise ValueError("'dist_ls' must be list type with one or more values representing 'distance to outfield wall'.")
        
        if len(angle_ls) != (len(dist_ls) + 1): 
            raise ValueError("'angle_ls' must have one more value than 'dist_ls' to satisfy sector requirements.")
        
        if not all(isinstance(phi, float) for phi in angle_ls):
            raise TypeError("All 'phi' values in 'angle_ls' must be of type float.")
            
        if not all(angle_ls[i] < angle_ls[i+1] for i in range(len(angle_ls) - 1)):
            raise ValueError("All 'phi' values in 'angle_ls' must be strictly increasing.")
        
        if not all(angle >= 0 for angle in angle_ls):
            raise ValueError("All 'phi' values in 'angle_ls' must be non-negative.")
        
        if not all(isinstance(dist, float) for dist in dist_ls):
            raise TypeError("All 'distance to outfield wall' values in 'dist_ls' must be of type float.")
        
        if not all(dist >= 0 for dist in dist_ls):
            raise ValueError("All 'distance to outfield wall' values in 'dist_ls' must be non-negative.")
        
        self.R = {(angle_ls[i], angle_ls[i+1]): dist_ls[i] for i in range(len(dist_ls))}

    def set_H(
        self, 
        angle_ls: list[float], 
        height_ls: list[float]
    ) -> None:
        """
        Sets the H dictionary, where keys are sectors represented by tuples of angles and
        values are the height of the outfield wall for that sector
        
        Args:
            angle_ls: list of angles, expects ordered
            dist_ls: list of heights, expects ordered
        """
        if not isinstance(angle_ls, list) or len(angle_ls) < 2:
            raise ValueError("'angle_ls' must be list type with two or more values representing heading angles 'phi'.")
        
        if not isinstance(height_ls, list) or len(height_ls) < 1:
            raise ValueError("'height_ls' must be list type with one or more values representing 'outfield wall height'.")
        
        if len(angle_ls) != (len(height_ls) + 1): 
            raise ValueError("'angle_ls' must have one more value than 'height_ls' to satisfy sector requirements.")
        
        if not all(isinstance(phi, float) for phi in angle_ls):
            raise TypeError("All 'phi' values in 'angle_ls' must be of type float.")

        if not all(angle_ls[i] < angle_ls[i+1] for i in range(len(angle_ls) - 1)):
            raise ValueError("All 'phi' values in 'angle_ls' must be strictly increasing.")
        
        if not all(angle >= 0 for angle in angle_ls):
            raise ValueError("All 'phi' values in 'angle_ls' must be non-negative.")
        
        if not all(isinstance(height, float) for height in height_ls):
            raise TypeError("All 'outfield wall height' values in 'height_ls' must be of type float.")
        
        if not all(dist >= 0 for dist in height_ls):
            raise ValueError("All 'distance to outfield wall' values in 'dist_ls' must be non-negative.")
        
        self.H = {(angle_ls[i], angle_ls[i+1]): height_ls[i] for i in range(len(height_ls))}

    def x(self, t) -> float:
        """x coordinate of position vector at time t"""
        return float(np.asarray(self._interp_x(t)).flat[0])

    def y(self, t) -> float:
        """y coordinate of position vector at time t"""
        return float(np.asarray(self._interp_y(t)).flat[0])

    def z(self, t) -> float:
        """z coordinate of position vector at time t"""
        return float(np.asarray(self._interp_z(t)).flat[0])
    
    def t_wall(self) -> float:
        """Solves for time, t, when the ball crosses the outfield wall"""
        if self.R is None:
            raise ValueError("Set R before proceeding.")
        
        if self._t_max is None:
            raise ValueError("Solve trajectory before proceeding.")
            
        target_dist = None
        for (low, high), dist in self.R.items():
            if low <= self.phi < high or (self.phi == high and self.phi == max(k[1] for k in self.R)):
                target_dist = dist
                break
            
        if target_dist is None:
            raise ValueError(f"Phi {self.phi} not covered in R sectors.")

        dist_func = lambda t: np.sqrt(self.x(t)**2 + self.y(t)**2) - target_dist
            
        # Initial guess based on horizontal velocity
        t_guess = target_dist / (self.s * np.cos(np.radians(self.theta)) + 1e-6)
        t_root = fsolve(dist_func, t_guess)[0]

        # Validate trajectory time range
        if t_root < 0 or t_root > self._t_max:
            return float('inf')
            
        return t_root
        
    def homerun_classifier(self) -> bool:
        """Classifies whether the batted ball is a home run"""
        if self.H is None:
            raise ValueError("Set H before proceeding.")
        
        t_w = self.t_wall()

        if t_w > self._t_max:
            return False

        model_height = self.z(t_w)

        if model_height <= 0.0:
            return False
        
        wall_height = None
        for (low, high), height in self.H.items():
            if low <= self.phi < high or (self.phi == high and self.phi == max(k[1] for k in self.H)):
                wall_height = height
                break

        if wall_height is None:
            raise ValueError(f"Phi {self.phi} not covered in H sectors.")

        return model_height > wall_height