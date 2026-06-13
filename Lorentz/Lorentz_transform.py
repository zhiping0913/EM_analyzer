import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple
import scipy.constants as C
class LorentzTransform:
    """
    A class for Lorentz transformations in special relativity.
    
    This class handles the transformation between inertial reference frames
    moving with constant relative velocity v = (vx, vy, vz).
    
    Attributes:
        v (np.ndarray): 3D velocity vector [vx, vy, vz] in m/s
        beta (np.ndarray): Velocity vector normalized by speed of light [vx/c, vy/c, vz/c]
        gamma (float): Lorentz factor γ = 1/√(1 - β²)
        transform_matrix (numpy.ndarray): 4×4 Lorentz transformation matrix
        inverse_matrix (numpy.ndarray): Pre-computed inverse transformation matrix
    """
    
    # Speed of light in m/s
    c = C.c
    
    # Vacuum permittivity and permeability
    epsilon0 = C.epsilon_0  # F/m
    mu0 = C.mu_0    # N/A²
    
    def __init__(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, 
                 beta_x: float = None, beta_y: float = None, beta_z: float = None,
                 c: float = None):
        """
        Initialize the Lorentz transformation with relative velocity.
        
        Parameters:
            vx (float): x-component of velocity in m/s (default: 0)
            vy (float): y-component of velocity in m/s (default: 0)
            vz (float): z-component of velocity in m/s (default: 0)
            beta_x (float): x-component of dimensionless velocity vx/c (optional)
            beta_y (float): y-component of dimensionless velocity vy/c (optional)
            beta_z (float): z-component of dimensionless velocity vz/c (optional)
            c (float): Speed of light in m/s (default: 299792458.0)
            
        Note:
            If beta_x, beta_y, beta_z are provided, they take precedence over vx, vy, vz.
            Otherwise, vx, vy, vz in m/s are converted to dimensionless beta = v/c.
        """
        # Override default speed of light if provided
        if c is not None:
            self.c = c
        
        # Determine whether to use beta or v
        if beta_x is not None or beta_y is not None or beta_z is not None:
            # Use provided beta values
            if beta_x is None or beta_y is None or beta_z is None:
                raise ValueError("If providing beta values, all three components must be provided")
            
            self.v = np.array([beta_x * self.c, beta_y * self.c, beta_z * self.c], dtype=np.float64)
            self.beta = np.array([beta_x, beta_y, beta_z], dtype=np.float64)
        else:
            # Use v values in m/s and convert to dimensionless beta
            self.v = np.array([vx, vy, vz], dtype=np.float64)  # m/s
            self.beta = self.v / self.c  # Dimensionless: v/c
        
        # Store original velocity magnitude
        self.v_norm = np.linalg.norm(self.v)
        self.beta_norm = np.linalg.norm(self.beta)
        
        # Check that velocity does not exceed speed of light
        if self.beta_norm >= 1.0:
            raise ValueError(
                f"Velocity magnitude ({self.v_norm:.2e} m/s = {self.beta_norm:.6f}c) "
                f"must be less than c={self.c:.2e} m/s"
            )
        
        # Calculate Lorentz factor γ = 1/√(1 - β²)
        self.gamma = 1.0 / np.sqrt(1.0 - self.beta_norm**2)
        
        # Generate the Lorentz transformation matrix
        self.transform_matrix = self._generate_matrix_hyperbolic()
        
        # Pre-compute inverse matrix for efficiency
        self.inverse_matrix = np.linalg.inv(self.transform_matrix)
        
        # Pre-compute velocity unit vector
        if self.beta_norm > 1e-10:
            self.beta_unit = self.beta / self.beta_norm
        else:
            self.beta_unit = np.zeros(3)
    
    @classmethod
    def from_beta(cls, beta_x: float = 0.0, beta_y: float = 0.0, beta_z: float = 0.0, 
                  c: float = None) -> 'LorentzTransform':
        """
        Alternative constructor using dimensionless beta directly.
        
        Parameters:
            beta_x (float): x-component of dimensionless velocity vx/c (default: 0)
            beta_y (float): y-component of dimensionless velocity vy/c (default: 0)
            beta_z (float): z-component of dimensionless velocity vz/c (default: 0)
            c (float): Speed of light in m/s (default: 299792458.0)
            
        Returns:
            LorentzTransform: Instance created from beta values
        """
        return cls(beta_x=beta_x, beta_y=beta_y, beta_z=beta_z, c=c)
    
    @classmethod
    def from_velocity(cls, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                      c: float = None) -> 'LorentzTransform':
        """
        Alternative constructor using velocity in m/s.
        
        Parameters:
            vx (float): x-component of velocity in m/s (default: 0)
            vy (float): y-component of velocity in m/s (default: 0)
            vz (float): z-component of velocity in m/s (default: 0)
            c (float): Speed of light in m/s (default: 299792458.0)
            
        Returns:
            LorentzTransform: Instance created from velocity values
        """
        return cls(vx=vx, vy=vy, vz=vz, c=c)
    
    @staticmethod
    def _transform(transform_matrix: np.ndarray, four_vector: np.ndarray) -> np.ndarray:
        """
        Pure function: Apply transformation matrix to a 4-vector.
        
        This is a stateless, pure function that performs only the mathematical
        transformation. It assumes inputs are already validated and in the
        correct format.
        
        Parameters:
            transform_matrix (numpy.ndarray): 4×4 transformation matrix
            four_vector (numpy.ndarray): 4-vector with shape (4, ...)
                where ... represents any number of additional dimensions
                
        Returns:
            numpy.ndarray: Transformed 4-vector with same shape as input
        """
        # Use einsum for efficient transformation of multi-dimensional arrays
        # Pattern: ij,j...->i... 
        # i,j are matrix indices (0-3), ... represents any extra dimensions
        return np.einsum('ij,j...->i...', transform_matrix, four_vector)
    
    @staticmethod
    def _em_fields_to_tensor(E: np.ndarray, B: np.ndarray, 
                            units: str = 'SI', 
                            c: float = 299792458.0,
                            epsilon0: float = 8.854187817e-12) -> np.ndarray:
        """
        Pure function: Convert electromagnetic fields to electromagnetic field tensor.
        
        Parameters:
            E (numpy.ndarray): Electric field with shape (3, ...)
            B (numpy.ndarray): Magnetic field with shape (3, ...)
            units (str): Units of input fields: 'SI', 'cgs', or 'natural'
            c (float): Speed of light in m/s
            epsilon0 (float): Vacuum permittivity in F/m
            
        Returns:
            numpy.ndarray: Electromagnetic field tensor with shape (4, 4, ...)
            
        Notes:
            The electromagnetic field tensor F^μν in natural units (c=1) is:
            
            F^μν = [ [0,  -Ex, -Ey, -Ez],
                     [Ex,  0,  -Bz,  By],
                     [Ey,  Bz,  0,  -Bx],
                     [Ez, -By,  Bx,  0] ]
            
            In SI units, the tensor components have different dimensions.
            We convert to natural units before constructing the tensor.
        """
        # Convert to natural units
        if units.upper() == 'SI':
            # SI to natural: E_nat = √(ε₀) * E_SI, B_nat = √(ε₀) * c * B_SI
            factor = np.sqrt(epsilon0)
            E_nat = factor * E
            B_nat = factor * c * B
        elif units.lower() == 'cgs':
            # CGS to natural: E_nat = E_cgs / √(4π), B_nat = B_cgs / √(4π)
            factor = 1.0 / np.sqrt(4.0 * np.pi)
            E_nat = factor * E
            B_nat = factor * B
        elif units.lower() == 'natural':
            # Already in natural units
            E_nat = E
            B_nat = B
        else:
            raise ValueError(f"Invalid units: {units}. Must be 'SI', 'cgs', or 'natural'")
        
        # Get shape information
        if E_nat.ndim == 1:
            # Single 3-vector
            Ex, Ey, Ez = E_nat
            Bx, By, Bz = B_nat
            # Create 4x4 tensor
            F = np.array([[0, -Ex, -Ey, -Ez],
                          [Ex, 0, -Bz, By],
                          [Ey, Bz, 0, -Bx],
                          [Ez, -By, Bx, 0]], dtype=np.float64)
        else:
            # Array with shape (3, ...)
            # Extract components
            Ex = E_nat[0]
            Ey = E_nat[1]
            Ez = E_nat[2]
            Bx = B_nat[0]
            By = B_nat[1]
            Bz = B_nat[2]
            
            # Get the shape of extra dimensions
            extra_shape = Ex.shape
            
            # Create electromagnetic field tensor with shape (4, 4, ...)
            F = np.zeros((4, 4) + extra_shape, dtype=np.float64)
            
            # Fill the tensor components
            # Row 0
            F[0, 1] = -Ex
            F[0, 2] = -Ey
            F[0, 3] = -Ez
            # Row 1
            F[1, 0] = Ex
            F[1, 2] = -Bz
            F[1, 3] = By
            # Row 2
            F[2, 0] = Ey
            F[2, 1] = Bz
            F[2, 3] = -Bx
            # Row 3
            F[3, 0] = Ez
            F[3, 1] = -By
            F[3, 2] = Bx
        
        # The tensor is antisymmetric: F[μ,ν] = -F[ν,μ]
        # We've already filled the upper triangular part (excluding diagonal)
        # Let's fill the lower triangular part
        for mu in range(4):
            for nu in range(mu+1, 4):
                F[nu, mu] = -F[mu, nu]
        
        return F
    
    @staticmethod
    def _tensor_to_em_fields(F: np.ndarray, 
                            units: str = 'SI',
                            c: float = 299792458.0,
                            epsilon0: float = 8.854187817e-12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure function: Extract electromagnetic fields from electromagnetic field tensor.
        
        Parameters:
            F (numpy.ndarray): Electromagnetic field tensor with shape (4, 4, ...)
            units (str): Units for output fields: 'SI', 'cgs', or 'natural'
            c (float): Speed of light in m/s
            epsilon0 (float): Vacuum permittivity in F/m
            
        Returns:
            tuple: (E, B) where E and B have shape (3, ...)
        """
        # Extract electric and magnetic fields from tensor in natural units
        if F.ndim == 2:
            # Single 4x4 tensor
            Ex_nat = F[1, 0]  # F^10
            Ey_nat = F[2, 0]  # F^20
            Ez_nat = F[3, 0]  # F^30
            
            Bx_nat = F[3, 2]  # F^32
            By_nat = F[1, 3]  # F^13
            Bz_nat = F[2, 1]  # F^21
        else:
            # Tensor with shape (4, 4, ...)
            Ex_nat = F[1, 0]
            Ey_nat = F[2, 0]
            Ez_nat = F[3, 0]
            
            Bx_nat = F[3, 2]
            By_nat = F[1, 3]
            Bz_nat = F[2, 1]
        
        # Stack into 3-vectors
        E_nat = np.stack([Ex_nat, Ey_nat, Ez_nat], axis=0)
        B_nat = np.stack([Bx_nat, By_nat, Bz_nat], axis=0)
        
        # Convert from natural units to desired units
        if units.upper() == 'SI':
            # Natural to SI: E_SI = E_nat / √(ε₀), B_SI = B_nat / (√(ε₀) * c)
            factor = 1.0 / np.sqrt(epsilon0)
            E = factor * E_nat
            B = factor * B_nat / c
        elif units.lower() == 'cgs':
            # Natural to CGS: E_cgs = E_nat * √(4π), B_cgs = B_nat * √(4π)
            factor = np.sqrt(4.0 * np.pi)
            E = factor * E_nat
            B = factor * B_nat
        elif units.lower() == 'natural':
            # Already in natural units
            E = E_nat
            B = B_nat
        else:
            raise ValueError(f"Invalid units: {units}. Must be 'SI', 'cgs', or 'natural'")
        
        return E, B
    
    @staticmethod
    def _transform_electromagnetic_tensor(transform_matrix: np.ndarray, 
                                         F: np.ndarray) -> np.ndarray:
        """
        Pure function: Transform electromagnetic field tensor using Lorentz transformation.
        
        The transformation formula is:
            F'^μν = Λ^μ_α Λ^ν_β F^αβ
        
        Parameters:
            transform_matrix (numpy.ndarray): 4×4 Lorentz transformation matrix
            F (numpy.ndarray): Electromagnetic field tensor with shape (4, 4, ...)
                
        Returns:
            numpy.ndarray: Transformed electromagnetic field tensor with same shape
        """
        # Use einsum for efficient tensor transformation
        # F'^μν = Λ^μ_α Λ^ν_β F^αβ
        # In Einstein notation: μν... = μα,νβ,αβ... -> μν...
        # where ... represents any extra dimensions
        return np.einsum('ia,jb,ab...->ij...', transform_matrix, transform_matrix, F)
    
    def _prepare_four_vector(self, four_vector: Union[list, tuple, np.ndarray], 
                           axis: int = 0) -> Tuple[np.ndarray, int]:
        """
        Prepare a four-vector for transformation.
        
        Converts input to numpy array, validates shape, and ensures
        4-vector components are along axis 0.
        
        Parameters:
            four_vector: Input 4-vector or array containing 4-vectors
            axis (int): Axis along which the 4-vector components are stored
            
        Returns:
            tuple: (prepared_array, original_axis)
        """
        # Convert to numpy array if not already
        if not isinstance(four_vector, np.ndarray):
            four_vector = np.asarray(four_vector, dtype=np.float64)
        
        # Validate shape
        if axis >= four_vector.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for array with {four_vector.ndim} dimensions")
        
        if four_vector.shape[axis] != 4:
            raise ValueError(f"Dimension along axis {axis} must be 4 for 4-vector, got {four_vector.shape[axis]}")
        
        # If components are not along axis 0, move them there
        if axis != 0:
            four_vector = np.moveaxis(four_vector, axis, 0)
            return four_vector, axis  # Return original axis for restoration
        
        return four_vector, 0
    
    def _prepare_three_vector(self, three_vector: Union[list, tuple, np.ndarray], 
                            axis: int = 0) -> Tuple[np.ndarray, int]:
        """
        Prepare a three-vector (e.g., velocity) for transformation.
        
        Converts input to numpy array, validates shape, and ensures
        3-vector components are along axis 0.
        
        Parameters:
            three_vector: Input 3-vector or array containing 3-vectors
            axis (int): Axis along which the 3-vector components are stored
            
        Returns:
            tuple: (prepared_array, original_axis)
        """
        # Convert to numpy array if not already
        if not isinstance(three_vector, np.ndarray):
            three_vector = np.asarray(three_vector, dtype=np.float64)
        
        # Validate shape
        if axis >= three_vector.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for array with {three_vector.ndim} dimensions")
        
        if three_vector.shape[axis] != 3:
            raise ValueError(f"Dimension along axis {axis} must be 3 for 3-vector, got {three_vector.shape[axis]}")
        
        # If components are not along axis 0, move them there
        if axis != 0:
            three_vector = np.moveaxis(three_vector, axis, 0)
            return three_vector, axis  # Return original axis for restoration
        
        return three_vector, 0
    
    def _restore_axis(self, transformed_vector: np.ndarray, 
                     original_shape: tuple, original_axis: int) -> np.ndarray:
        """
        Restore the transformed array to its original axis orientation.
        
        Parameters:
            transformed_vector: Array after transformation
            original_shape: Original shape before moving axis
            original_axis: Original axis where vector components were stored
            
        Returns:
            numpy.ndarray: Array with original axis restored
        """
        if original_axis != 0:
            # Move axis 0 back to its original position
            return np.moveaxis(transformed_vector, 0, original_axis)
        return transformed_vector
    
    def transform(self, four_vector: Union[list, tuple, np.ndarray], 
                  direction: str = '0->1', axis: int = 0) -> np.ndarray:
        """
        Apply Lorentz transformation to a 4-vector with specified direction.
        
        Parameters:
            four_vector: 4-vector components
                Can be:
                - Simple 4-vector: [t, x, y, z] (shape: (4,))
                - Array with extra dimensions: shape (4, ...)
                - Array with 4-vector components along specified axis
            direction (str): Direction of transformation:
                '0->1' or 'forward': Transform from frame 0 to frame 1
                '1->0' or 'backward': Transform from frame 1 to frame 0
                'forward' and 'backward' are aliases for '0->1' and '1->0'
            axis (int): Axis along which the 4-vector components are stored
                
        Returns:
            numpy.ndarray: Transformed 4-vector with same shape and axis as input
            
        Raises:
            ValueError: If direction is not recognized or four_vector is invalid
        """
        # Prepare the four-vector (validate and move components to axis 0)
        prepared_vector, original_axis = self._prepare_four_vector(four_vector, axis)
        original_shape = prepared_vector.shape
        
        # Select the appropriate transformation matrix
        matrix = self._select_matrix(direction)
        
        # Apply the pure transformation
        transformed = self._transform(matrix, prepared_vector)
        
        # Restore the original axis orientation
        return self._restore_axis(transformed, original_shape, original_axis)
    
    def transform_velocity(self, velocity: Union[list, tuple, np.ndarray], 
                          direction: str = '0->1', axis: int = 0,
                          input_units: str = 'dimensionless',
                          output_units: str = 'dimensionless') -> np.ndarray:
        """
        Apply relativistic velocity addition formula to a velocity field.
        
        Transforms velocities between inertial frames using the relativistic
        velocity addition formula.
        
        Parameters:
            velocity: Velocity 3-vector or field
                Can be:
                - Simple 3-vector: [vx, vy, vz]
                - Array with extra dimensions: shape (3, ...)
                - Array with 3-vector components along specified axis
            direction (str): Direction of transformation:
                '0->1' or 'forward': Transform velocities from frame 0 to frame 1
                '1->0' or 'backward': Transform velocities from frame 1 to frame 0
            axis (int): Axis along which the velocity components are stored
            input_units (str): Units of input velocity:
                'dimensionless' or 'beta': v/c (default)
                'm/s': velocity in meters per second
            output_units (str): Units of output velocity:
                'dimensionless' or 'beta': v/c (default)
                'm/s': velocity in meters per second
                
        Returns:
            numpy.ndarray: Transformed velocity with same shape and axis as input
            
        Notes:
            Uses the relativistic velocity addition formula.
            All internal calculations are done in dimensionless units (beta).
        """
        # Prepare the velocity (validate and move components to axis 0)
        prepared_velocity, original_axis = self._prepare_three_vector(velocity, axis)
        original_shape = prepared_velocity.shape
        
        # Convert input to dimensionless beta if needed
        if input_units.lower() in ['m/s', 'meters', 'meters_per_second']:
            # Convert from m/s to dimensionless beta
            prepared_velocity = prepared_velocity / self.c
        
        # Get transformation parameters
        if direction.lower() in ['0->1', 'forward']:
            V = self.beta  # Frame 1 beta relative to frame 0
            gamma = self.gamma
        elif direction.lower() in ['1->0', 'backward']:
            V = -self.beta  # For backward transform, use negative beta
            gamma = self.gamma
        else:
            raise ValueError(f"Unknown direction: {direction}. Use '0->1', '1->0', 'forward', or 'backward'")
        
        # Reshape for broadcasting
        V = V.reshape(3, 1)  # Shape (3, 1) for broadcasting
        
        # Extract velocity components
        vx = prepared_velocity[0:1, ...]  # Keep dimensions for broadcasting
        vy = prepared_velocity[1:2, ...]
        vz = prepared_velocity[2:3, ...]
        
        # Calculate dot product v·V for all elements
        v_dot_V = vx * V[0] + vy * V[1] + vz * V[2]  # Shape (1, ...)
        
        # Avoid division by zero or very small numbers
        denominator = 1.0 + v_dot_V
        
        # For backward transformation
        if direction.lower() in ['1->0', 'backward']:
            denominator = 1.0 - v_dot_V
        
        # Handle the case where V is zero (no transformation needed)
        if np.allclose(V, 0.0):
            transformed = prepared_velocity
        else:
            # Calculate parallel and perpendicular components
            # Parallel component magnitude
            v_parallel_mag = v_dot_V / self.beta_norm if self.beta_norm > 1e-10 else 0.0
            
            # Unit vector in direction of V
            if self.beta_norm > 1e-10:
                n = V / self.beta_norm
            else:
                n = np.zeros((3, 1))
            
            # Parallel component vector
            v_parallel = v_parallel_mag * n  # Shape (3, ...)
            
            # Perpendicular component vector
            v_perp = prepared_velocity - v_parallel  # Shape (3, ...)
            
            # Apply relativistic velocity addition formula
            if direction.lower() in ['0->1', 'forward']:
                transformed = (v_parallel + V) / denominator + v_perp / (gamma * denominator)
            else:  # '1->0' or 'backward'
                transformed = (v_parallel - V) / denominator + v_perp / (gamma * denominator)
        
        # Convert output to desired units
        if output_units.lower() in ['m/s', 'meters', 'meters_per_second']:
            transformed = transformed * self.c
        
        # Restore the original axis orientation
        return self._restore_axis(transformed, original_shape, original_axis)
    
    def transform_field(self, component0: Union[float, np.ndarray],
                       component1: np.ndarray, 
                       component2: np.ndarray, 
                       component3: np.ndarray,
                       four_vector_type: str = 'spacetime',
                       direction: str = '0->1',
                       input_units: str = 'SI',
                       output_units: str = 'SI') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform a 4-vector field defined on a spatial grid.
        
        Parameters:
            component0: Zeroth component of the 4-vector
                - For 'spacetime': time coordinate t (seconds)
                - For 'four_momentum': energy E (joules)
                - For 'four_current': charge density ρ (coulombs/m³)
                - For 'four_wave': angular frequency ω (radians/second)
            component1: First component of the 4-vector
                - For 'spacetime': x-coordinate (meters)
                - For 'four_momentum': x-momentum p_x (kg·m/s)
                - For 'four_current': x-current density j_x (amperes/m²)
                - For 'four_wave': x-wave number k_x (radians/meter)
            component2: Second component of the 4-vector (same units as component1)
            component3: Third component of the 4-vector (same units as component1)
            four_vector_type (str): Type of 4-vector:
                'spacetime': (ct, x, y, z) [default]
                'four_momentum': (E/c, p_x, p_y, p_z)
                'four_current': (ρc, j_x, j_y, j_z)
                'four_wave': (ω/c, k_x, k_y, k_z)
            direction (str): Direction of transformation:
                '0->1' or 'forward': Transform from frame 0 to frame 1
                '1->0' or 'backward': Transform from frame 1 to frame 0
            input_units (str): Units of input components:
                'SI': International System of Units (default)
                'natural': Natural units where c=1
            output_units (str): Units of output components:
                'SI': International System of Units (default)
                'natural': Natural units where c=1
                
        Returns:
            tuple: (transformed_component0, transformed_component1, 
                    transformed_component2, transformed_component3)
                   in the same units as specified by output_units
                   
        Notes:
            The Lorentz transformation matrix works on 4-vectors in natural units.
            For proper transformation, we need to convert SI units to natural units,
            apply the transformation, then convert back if needed.
            
            Conversion rules (SI to natural):
            - spacetime: t -> ct (multiply by c), x,y,z remain the same
            - four_momentum: E -> E/c (divide by c), p remain the same
            - four_current: ρ -> ρc (multiply by c), j remain the same
            - four_wave: ω -> ω/c (divide by c), k remain the same
            
            In natural units, all components have the same dimension (length).
        """
        # Validate field shapes
        shapes = [np.asarray(comp).shape for comp in [component1, component2, component3] ]
        
        if len(shapes) > 0:
            # Check that all spatial components have the same shape
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("All spatial components must have the same shape")
            
            spatial_shape = shapes[0]
            
            # Handle scalar component0
            if np.isscalar(component0):
                # Scalar: create array with same shape as spatial components
                component0_array = np.full(spatial_shape, component0, dtype=np.float64)
            else:
                # Array: check shape
                component0_array = np.asarray(component0, dtype=np.float64)
                if component0_array.shape != spatial_shape:
                    raise ValueError(f"component0 shape {component0_array.shape} must match spatial components shape {spatial_shape}")
        else:
            # All components are scalars
            component0_array = np.asarray(component0, dtype=np.float64)
            component1 = np.asarray(component1, dtype=np.float64)
            component2 = np.asarray(component2, dtype=np.float64)
            component3 = np.asarray(component3, dtype=np.float64)
            
            # Create 0-dimensional arrays
            component0_array = component0_array.reshape(())
            component1 = component1.reshape(())
            component2 = component2.reshape(())
            component3 = component3.reshape(())
        
        # Define conversion factors for different 4-vector types
        # (factor0, factor1, factor2, factor3)
        # For conversion from SI to natural units
        conversion_to_natural = {
            'spacetime': (self.c, 1.0, 1.0, 1.0),      # t -> ct
            'four_momentum': (1.0/self.c, 1.0, 1.0, 1.0),  # E -> E/c
            'four_current': (self.c, 1.0, 1.0, 1.0),   # ρ -> ρc
            'four_wave': (1.0/self.c, 1.0, 1.0, 1.0),  # ω -> ω/c
        }
        
        # For conversion from natural units back to SI
        conversion_to_SI = {
            'spacetime': (1.0/self.c, 1.0, 1.0, 1.0),  # ct -> t
            'four_momentum': (self.c, 1.0, 1.0, 1.0),  # E/c -> E
            'four_current': (1.0/self.c, 1.0, 1.0, 1.0),  # ρc -> ρ
            'four_wave': (self.c, 1.0, 1.0, 1.0),      # ω/c -> ω
        }
        
        # Validate four_vector_type
        if four_vector_type not in conversion_to_natural:
            valid_types = list(conversion_to_natural.keys())
            raise ValueError(f"Invalid four_vector_type: {four_vector_type}. Must be one of {valid_types}")
        
        # Convert to natural units if input is in SI units
        if input_units.upper() == 'SI':
            factor0, factor1, factor2, factor3 = conversion_to_natural[four_vector_type]
            component0_natural = component0_array * factor0
            component1_natural = component1 * factor1
            component2_natural = component2 * factor2
            component3_natural = component3 * factor3
        elif input_units.lower() == 'natural':
            # Already in natural units
            component0_natural = component0_array
            component1_natural = component1
            component2_natural = component2
            component3_natural = component3
        else:
            raise ValueError(f"Invalid input_units: {input_units}. Must be 'SI' or 'natural'")
        
        # Stack components along a new axis (axis 0)
        components_natural = np.stack([component0_natural, component1_natural, 
                                       component2_natural, component3_natural], axis=0)
        
        # Apply transformation (components are along axis 0)
        transformed_natural = self.transform(components_natural, direction, axis=0)
        
        # Extract components
        component0_trans_natural = transformed_natural[0]
        component1_trans_natural = transformed_natural[1]
        component2_trans_natural = transformed_natural[2]
        component3_trans_natural = transformed_natural[3]
        
        # Convert back from natural units if output is in SI units
        if output_units.upper() == 'SI':
            factor0, factor1, factor2, factor3 = conversion_to_SI[four_vector_type]
            component0_trans = component0_trans_natural * factor0
            component1_trans = component1_trans_natural * factor1
            component2_trans = component2_trans_natural * factor2
            component3_trans = component3_trans_natural * factor3
        elif output_units.lower() == 'natural':
            # Keep in natural units
            component0_trans = component0_trans_natural
            component1_trans = component1_trans_natural
            component2_trans = component2_trans_natural
            component3_trans = component3_trans_natural
        else:
            raise ValueError(f"Invalid output_units: {output_units}. Must be 'SI' or 'natural'")
        
        return component0_trans, component1_trans, component2_trans, component3_trans
    
    def transform_electromagnetic_field(self, E_field: Union[list, tuple, np.ndarray], 
                                       B_field: Union[list, tuple, np.ndarray],
                                       direction: str = '0->1',
                                       axis: int = 0,
                                       input_units: str = 'SI',
                                       output_units: str = 'SI') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform electromagnetic fields between inertial frames.
        
        Parameters:
            E_field: Electric field 3-vector or field
                Can be:
                - Simple 3-vector: [E_x, E_y, E_z]
                - Array with extra dimensions: shape (3, ...)
                - Array with 3-vector components along specified axis
            B_field: Magnetic field 3-vector or field
                Same shape requirements as E_field
            direction (str): Direction of transformation:
                '0->1' or 'forward': Transform from frame 0 to frame 1
                '1->0' or 'backward': Transform from frame 1 to frame 0
            axis (int): Axis along which the field components are stored
            input_units (str): Units of input fields:
                'SI': V/m for E, Tesla for B (default)
                'cgs': statvolt/cm for E, Gauss for B
                'natural': Natural units where E and B have same units
            output_units (str): Units of output fields (same options as input_units)
                
        Returns:
            tuple: (E_transformed, B_transformed) with same shape and units as specified
            
        Notes:
            Uses electromagnetic field tensor transformation formula:
            F'^μν = Λ^μ_α Λ^ν_β F^αβ
        """
        # Prepare the fields (validate and move components to axis 0)
        E_prepared, original_axis = self._prepare_three_vector(E_field, axis)
        B_prepared, _ = self._prepare_three_vector(B_field, axis)
        
        original_shape = E_prepared.shape
        
        # Validate that E and B have the same shape
        if E_prepared.shape != B_prepared.shape:
            raise ValueError(f"E_field and B_field must have the same shape. "
                           f"Got E: {E_prepared.shape}, B: {B_prepared.shape}")
        
        # Get the transformation matrix
        if direction.lower() in ['0->1', 'forward']:
            transform_matrix = self.transform_matrix
        elif direction.lower() in ['1->0', 'backward']:
            transform_matrix = self.inverse_matrix
        else:
            raise ValueError(f"Unknown direction: {direction}. Use '0->1', '1->0', 'forward', or 'backward'")
        
        # Convert electromagnetic fields to tensor using pure function
        F = self._em_fields_to_tensor(
            E_prepared, B_prepared, 
            units=input_units,
            c=self.c,
            epsilon0=self.epsilon0
        )
        
        # Transform the tensor using pure function
        F_transformed = self._transform_electromagnetic_tensor(transform_matrix, F)
        
        # Extract transformed fields from tensor using pure function
        E_transformed, B_transformed = self._tensor_to_em_fields(
            F_transformed,
            units=output_units,
            c=self.c,
            epsilon0=self.epsilon0
        )
        
        # Restore the original axis orientation
        E_transformed = self._restore_axis(E_transformed, original_shape, original_axis)
        B_transformed = self._restore_axis(B_transformed, original_shape, original_axis)
        
        return E_transformed, B_transformed
    
    def _select_matrix(self, direction: str) -> np.ndarray:
        """
        Select the appropriate transformation matrix based on direction.
        
        Parameters:
            direction (str): Direction of transformation
            
        Returns:
            numpy.ndarray: Transformation matrix
        """
        direction = direction.lower()
        if direction in ['0->1', 'forward']:
            return self.transform_matrix
        elif direction in ['1->0', 'backward']:
            return self.inverse_matrix
        else:
            raise ValueError(f"Unknown direction: {direction}. Use '0->1', '1->0', 'forward', or 'backward'")
    
    def inverse_transform(self, four_vector: Union[list, tuple, np.ndarray], 
                         axis: int = 0) -> np.ndarray:
        """
        Apply inverse Lorentz transformation to a 4-vector.
        
        Parameters:
            four_vector: 4-vector in any valid format
            axis (int): Axis along which the 4-vector components are stored
            
        Returns:
            numpy.ndarray: Original 4-vector before transformation
        """
        return self.transform(four_vector, direction='1->0', axis=axis)
    
    def inverse_transform_velocity(self, velocity: Union[list, tuple, np.ndarray], 
                                 axis: int = 0,
                                 input_units: str = 'dimensionless',
                                 output_units: str = 'dimensionless') -> np.ndarray:
        """
        Apply inverse velocity transformation.
        
        Parameters:
            velocity: Velocity in frame 1
            axis (int): Axis along which the velocity components are stored
            input_units (str): Units of input velocity
            output_units (str): Units of output velocity
            
        Returns:
            numpy.ndarray: Velocity in frame 0
        """
        return self.transform_velocity(velocity, direction='1->0', axis=axis,
                                       input_units=input_units, output_units=output_units)
    
    def inverse_transform_electromagnetic_field(self, E_field: Union[list, tuple, np.ndarray], 
                                               B_field: Union[list, tuple, np.ndarray],
                                               axis: int = 0,
                                               input_units: str = 'SI',
                                               output_units: str = 'SI') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply inverse electromagnetic field transformation.
        
        Parameters:
            E_field: Electric field in frame 1
            B_field: Magnetic field in frame 1
            axis (int): Axis along which the field components are stored
            input_units (str): Units of input fields
            output_units (str): Units of output fields
            
        Returns:
            tuple: (E, B) in frame 0
        """
        return self.transform_electromagnetic_field(
            E_field, B_field, 
            direction='1->0',
            axis=axis,
            input_units=input_units,
            output_units=output_units
        )
    
    def _generate_matrix_hyperbolic(self) -> np.ndarray:
        """
        Generate the 4×4 Lorentz transformation matrix using hyperbolic functions.
        
        Returns:
            numpy.ndarray: 4×4 Lorentz transformation matrix
        """
        # Initialize a 4x4 identity matrix
        lambda_matrix = np.identity(4, dtype=np.float64)
        
        # If velocity is zero, return identity matrix (no transformation)
        if np.allclose(self.beta, 0.0):
            return lambda_matrix
        
        # Extract components
        beta_x, beta_y, beta_z = self.beta
        beta_norm = self.beta_norm
        
        # Unit vector in the direction of velocity
        n_x = beta_x / beta_norm
        n_y = beta_y / beta_norm
        n_z = beta_z / beta_norm
        
        # Rapidity ξ = atanh(β)
        xi = np.arctanh(beta_norm)
        
        # Hyperbolic functions of rapidity
        cosh_xi = np.cosh(xi)  # This equals γ
        sinh_xi = np.sinh(xi)  # This equals γβ
        
        # Time-time component: Λ⁰₀ = γ = cosh(ξ)
        lambda_matrix[0, 0] = cosh_xi
        
        # Time-space components: Λ⁰_i = -γβ_i = -sinh(ξ) * n_i
        lambda_matrix[0, 1] = -sinh_xi * n_x
        lambda_matrix[0, 2] = -sinh_xi * n_y
        lambda_matrix[0, 3] = -sinh_xi * n_z
        
        # Space-time components: Λⁱ₀ = -γβ_i = -sinh(ξ) * n_i
        lambda_matrix[1, 0] = -sinh_xi * n_x
        lambda_matrix[2, 0] = -sinh_xi * n_y
        lambda_matrix[3, 0] = -sinh_xi * n_z
        
        # Space-space components: Λⁱ_j = δⁱ_j + (cosh(ξ)-1)n_i n_j
        cosh_minus_1 = cosh_xi - 1.0
        
        for i in range(1, 4):
            for j in range(1, 4):
                if i == 1:
                    n_i = n_x
                elif i == 2:
                    n_i = n_y
                else:  # i == 3
                    n_i = n_z
                    
                if j == 1:
                    n_j = n_x
                elif j == 2:
                    n_j = n_y
                else:  # j == 3
                    n_j = n_z
                
                lambda_matrix[i, j] = (1.0 if i == j else 0.0) + cosh_minus_1 * n_i * n_j
        
        return lambda_matrix
    
    def transform_velocity_field(self, vx_field: np.ndarray, vy_field: np.ndarray, vz_field: np.ndarray,
                                direction: str = '0->1',
                                input_units: str = 'm/s',
                                output_units: str = 'm/s') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform a velocity field defined on a spatial grid.
        
        Parameters:
            vx_field (numpy.ndarray): x-velocity component field
            vy_field (numpy.ndarray): y-velocity component field  
            vz_field (numpy.ndarray): z-velocity component field
            direction (str): Direction of transformation
            input_units (str): Units of input velocity
            output_units (str): Units of output velocity
            
        Returns:
            tuple: (vx_transformed, vy_transformed, vz_transformed)
        """
        # Validate field shapes
        if vx_field.shape != vy_field.shape or vx_field.shape != vz_field.shape:
            raise ValueError("All velocity field components must have the same shape")
        
        # Stack components along a new axis (axis 0)
        velocity_components = np.stack([vx_field, vy_field, vz_field], axis=0)
        
        # Apply velocity transformation
        transformed = self.transform_velocity(velocity_components, direction, axis=0,
                                             input_units=input_units, output_units=output_units)
        
        # Extract components
        vx_transformed = transformed[0]
        vy_transformed = transformed[1]
        vz_transformed = transformed[2]
        
        return vx_transformed, vy_transformed, vz_transformed
    
    def transform_batch(self, four_vectors: np.ndarray, 
                       direction: str = '0->1', axis: int = 0) -> np.ndarray:
        """
        Transform a batch of 4-vectors.
        
        Parameters:
            four_vectors (numpy.ndarray): Array containing multiple 4-vectors
            direction (str): Direction of transformation
            axis (int): Axis along which the 4-vector components are stored
                
        Returns:
            numpy.ndarray: Transformed 4-vectors
        """
        return self.transform(four_vectors, direction, axis)
    
    def transform_velocity_batch(self, velocities: np.ndarray,
                               direction: str = '0->1', axis: int = 0,
                               input_units: str = 'dimensionless',
                               output_units: str = 'dimensionless') -> np.ndarray:
        """
        Transform a batch of velocities.
        
        Parameters:
            velocities (numpy.ndarray): Array containing multiple velocity 3-vectors
            direction (str): Direction of transformation
            axis (int): Axis along which the velocity components are stored
            input_units (str): Units of input velocity
            output_units (str): Units of output velocity
                
        Returns:
            numpy.ndarray: Transformed velocities
        """
        return self.transform_velocity(velocities, direction, axis, 
                                       input_units=input_units, output_units=output_units)
    
    def transform_electromagnetic_field_batch(self, E_fields: np.ndarray, B_fields: np.ndarray,
                                            direction: str = '0->1', axis: int = 0,
                                            input_units: str = 'SI',
                                            output_units: str = 'SI') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a batch of electromagnetic fields.
        
        Parameters:
            E_fields (numpy.ndarray): Electric field batch
            B_fields (numpy.ndarray): Magnetic field batch
            direction (str): Direction of transformation
            axis (int): Axis along which the field components are stored
            input_units (str): Units of input fields
            output_units (str): Units of output fields
                
        Returns:
            tuple: (E_transformed, B_transformed)
        """
        # This is essentially the same as transform_electromagnetic_field
        # but with a different name for clarity when dealing with batches
        return self.transform_electromagnetic_field(
            E_fields, B_fields, direction, axis, input_units, output_units
        )
    
    # Additional utility methods
    def get_velocity(self, units: str = 'm/s') -> np.ndarray:
        """Get the velocity vector.
        
        Parameters:
            units (str): Units of output:
                'm/s': velocity in meters per second (default)
                'beta' or 'dimensionless': v/c
        
        Returns:
            numpy.ndarray: Velocity vector
        """
        if units.lower() in ['beta', 'dimensionless']:
            return self.beta.copy()
        else:  # 'm/s' or any other
            return self.v.copy()
    
    def get_gamma(self) -> float:
        """Get the Lorentz factor."""
        return self.gamma
    
    def get_rapidity(self) -> float:
        """Get the rapidity ξ = atanh(β)."""
        if self.beta_norm < 1e-10:
            return 0.0
        return np.arctanh(self.beta_norm)
    
    def get_inverse_matrix(self) -> np.ndarray:
        """Get the inverse Lorentz transformation matrix."""
        return self.inverse_matrix
    
    def compose(self, other: 'LorentzTransform') -> 'LorentzTransform':
        """
        Compose two Lorentz transformations.
        
        Parameters:
            other (LorentzTransform): Another Lorentz transformation
            
        Returns:
            LorentzTransform: The composed transformation
        """
        # Check that both instances use the same speed of light
        if abs(self.c - other.c) > 1e-6:
            raise ValueError(f"Cannot compose transforms with different c values: {self.c} vs {other.c}")
        
        # The composed matrix is the product of the two matrices
        product_matrix = np.dot(self.transform_matrix, other.transform_matrix)
        
        # Extract beta from the product matrix:
        # β_i = -Λ⁰_i / Λ⁰₀
        beta_x = -product_matrix[0, 1] / product_matrix[0, 0]
        beta_y = -product_matrix[0, 2] / product_matrix[0, 0]
        beta_z = -product_matrix[0, 3] / product_matrix[0, 0]
        
        return LorentzTransform.from_beta(beta_x, beta_y, beta_z, c=self.c)
    
    def __str__(self) -> str:
        """String representation of the Lorentz transformation."""
        rapidity = self.get_rapidity()
        return (f"LorentzTransform(v=[{self.v[0]:.2e}, {self.v[1]:.2e}, {self.v[2]:.2e}] m/s, "
                f"β=[{self.beta[0]:.6f}, {self.beta[1]:.6f}, {self.beta[2]:.6f}], "
                f"|β|={self.beta_norm:.6f}, γ={self.gamma:.4f}, ξ={rapidity:.4f}, "
                f"c={self.c:.2e} m/s)")


# 独立的纯函数
def em_fields_to_tensor(E: np.ndarray, B: np.ndarray, 
                       units: str = 'SI', 
                       c: float = 299792458.0,
                       epsilon0: float = 8.854187817e-12) -> np.ndarray:
    """
    纯函数：将电磁场转换为电磁场张量。
    
    Parameters:
        E (numpy.ndarray): 电场，形状 (3, ...)
        B (numpy.ndarray): 磁场，形状 (3, ...)
        units (str): 输入场单位：'SI', 'cgs', 或 'natural'
        c (float): 光速，单位 m/s
        epsilon0 (float): 真空介电常数，单位 F/m
        
    Returns:
        numpy.ndarray: 电磁场张量，形状 (4, 4, ...)
    """
    return LorentzTransform._em_fields_to_tensor(E, B, units, c, epsilon0)


def tensor_to_em_fields(F: np.ndarray, 
                       units: str = 'SI',
                       c: float = 299792458.0,
                       epsilon0: float = 8.854187817e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    纯函数：从电磁场张量提取电磁场。
    
    Parameters:
        F (numpy.ndarray): 电磁场张量，形状 (4, 4, ...)
        units (str): 输出场单位：'SI', 'cgs', 或 'natural'
        c (float): 光速，单位 m/s
        epsilon0 (float): 真空介电常数，单位 F/m
        
    Returns:
        tuple: (E, B)，其中 E 和 B 形状为 (3, ...)
    """
    return LorentzTransform._tensor_to_em_fields(F, units, c, epsilon0)


def transform_electromagnetic_tensor(transform_matrix: np.ndarray, 
                                    F: np.ndarray) -> np.ndarray:
    """
    纯函数：使用洛伦兹变换矩阵变换电磁场张量。
    
    Parameters:
        transform_matrix (numpy.ndarray): 4×4 洛伦兹变换矩阵
        F (numpy.ndarray): 电磁场张量，形状 (4, 4, ...)
            
    Returns:
        numpy.ndarray: 变换后的电磁场张量，形状相同
    """
    return LorentzTransform._transform_electromagnetic_tensor(transform_matrix, F)


# 测试电磁场张量变换
if __name__ == "__main__":
    print("Testing Electromagnetic Tensor Transformation")
    print("=" * 70)
    
    # 创建洛伦兹变换
    lorentz = LorentzTransform.from_beta(beta_x=0.8, beta_y=0.2, beta_z=0.1)
    
    # 测试纯函数：电磁场到张量转换
    print("\nTest 1: Pure function - EM fields to tensor")
    
    # 简单的电磁场
    E = np.array([1.0, 2.0, 3.0])
    B = np.array([0.1, 0.2, 0.3])
    
    # 转换为张量
    F = em_fields_to_tensor(E, B, units='SI', c=lorentz.c, epsilon0=lorentz.epsilon0)
    
    print(f"Electric field: {E}")
    print(f"Magnetic field: {B}")
    print(f"Electromagnetic field tensor (F^μν):")
    print(F)
    
    # 测试张量到电磁场转换
    print("\nTest 2: Pure function - Tensor to EM fields")
    
    E_back, B_back = tensor_to_em_fields(F, units='SI', c=lorentz.c, epsilon0=lorentz.epsilon0)
    
    print(f"Recovered electric field: {E_back}")
    print(f"Recovered magnetic field: {B_back}")
    print(f"Fields match original: {np.allclose(E, E_back) and np.allclose(B, B_back)}")
    
    # 测试电磁场张量变换
    print("\nTest 3: Pure function - EM tensor transformation")
    
    # 变换张量
    F_transformed = transform_electromagnetic_tensor(lorentz.transform_matrix, F)
    
    print(f"Original tensor F^μν:")
    print(F)
    print(f"\nTransformed tensor F'^μν:")
    print(F_transformed)
    
    # 从变换后的张量提取电磁场
    E_transformed, B_transformed = tensor_to_em_fields(
        F_transformed, units='SI', c=lorentz.c, epsilon0=lorentz.epsilon0
    )
    
    print(f"\nTransformed electric field: {E_transformed}")
    print(f"Transformed magnetic field: {B_transformed}")
    
    # 与类方法结果比较
    print("\nTest 4: Comparison with class method")
    
    E_class, B_class = lorentz.transform_electromagnetic_field(
        E, B, direction='0->1', input_units='SI', output_units='SI'
    )
    
    print(f"Class method - E: {E_class}")
    print(f"Class method - B: {B_class}")
    print(f"Pure function - E: {E_transformed}")
    print(f"Pure function - B: {B_transformed}")
    print(f"Results match: {np.allclose(E_class, E_transformed) and np.allclose(B_class, B_transformed)}")
    
    # 测试多维数组
    print("\nTest 5: Multi-dimensional arrays")
    
    # 创建形状为 (3, 2, 3) 的电磁场
    E_grid = np.random.randn(3, 2, 3)
    B_grid = np.random.randn(3, 2, 3)
    
    print(f"E grid shape: {E_grid.shape}")
    print(f"B grid shape: {B_grid.shape}")
    
    # 转换为张量
    F_grid = em_fields_to_tensor(
        E_grid, B_grid, units='SI', c=lorentz.c, epsilon0=lorentz.epsilon0
    )
    
    print(f"F tensor shape: {F_grid.shape}")
    
    # 变换张量
    F_grid_transformed = transform_electromagnetic_tensor(lorentz.transform_matrix, F_grid)
    
    print(f"Transformed F tensor shape: {F_grid_transformed.shape}")
    
    # 提取变换后的场
    E_grid_transformed, B_grid_transformed = tensor_to_em_fields(
        F_grid_transformed, units='SI', c=lorentz.c, epsilon0=lorentz.epsilon0
    )
    
    print(f"Transformed E grid shape: {E_grid_transformed.shape}")
    print(f"Transformed B grid shape: {B_grid_transformed.shape}")
    
    # 与类方法比较
    E_grid_class, B_grid_class = lorentz.transform_electromagnetic_field(
        E_grid, B_grid, direction='0->1', axis=0, input_units='SI', output_units='SI'
    )
    
    print(f"\nClass method shapes - E: {E_grid_class.shape}, B: {B_grid_class.shape}")
    print(f"Pure function shapes - E: {E_grid_transformed.shape}, B: {B_grid_transformed.shape}")
    print(f"Results match: {np.allclose(E_grid_class, E_grid_transformed) and np.allclose(B_grid_class, B_grid_transformed)}")
    
    # 测试洛伦兹不变量
    print("\nTest 6: Lorentz invariants with tensor method")
    
    # 计算不变量
    # 1. E·B
    # 2. E² - c²B²
    
    # 对于张量，不变量可以通过张量计算：
    # 第一个不变量: (1/2)F_μνF^μν = B² - E² (在自然单位中)
    # 第二个不变量: (1/4)F_μν(*F)^μν = E·B
    
    # 直接计算电磁场的不变量
    E_dot_B = np.sum(E * B)
    E_sq_minus_c2B_sq = np.sum(E**2) - (lorentz.c**2) * np.sum(B**2)
    
    print(f"In frame 0:")
    print(f"  E·B = {E_dot_B:.6f}")
    print(f"  E² - c²B² = {E_sq_minus_c2B_sq:.6f}")
    
    # 变换后的不变量
    E_dot_B_trans = np.sum(E_transformed * B_transformed)
    E_sq_minus_c2B_sq_trans = np.sum(E_transformed**2) - (lorentz.c**2) * np.sum(B_transformed**2)
    
    print(f"\nIn frame 1 (transformed):")
    print(f"  E·B = {E_dot_B_trans:.6f}")
    print(f"  E² - c²B² = {E_sq_minus_c2B_sq_trans:.6f}")
    
    print(f"\nInvariants should be preserved:")
    print(f"  E·B preserved: {np.abs(E_dot_B - E_dot_B_trans) < 1e-10}")
    print(f"  E² - c²B² preserved: {np.abs(E_sq_minus_c2B_sq - E_sq_minus_c2B_sq_trans) < 1e-10}")