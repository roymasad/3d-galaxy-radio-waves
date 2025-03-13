import numpy as np
import random
from star import Star


class GalaxyModel:
    """
    Represents a 3D galaxy model with stars.
    
    The galaxy is modeled as a spiral structure with multiple arms and a central bulge.
    """
    def __init__(self, diameter=2000, num_stars=500, num_radio_emitting=300,
                 radio_signal_duration=10_000, radio_civilization_lifetime=100, 
                 radio_blast_duration=10, simulation_duration=1_000_000,
                 num_arms=3, arm_width=0.3, arm_tightness=0.3, bulge_size=0.0):
        """
        Initialize the galaxy model.
        
        Args:
            diameter: Galaxy diameter in light-years
            num_stars: Total number of stars in the galaxy
            num_radio_emitting: Number of stars that emit radio waves
            radio_signal_duration: Duration of radio signal emission in years
            radio_civilization_lifetime: How long the civilization emits radio waves
            radio_blast_duration: Duration of a single radio burst in years
            simulation_duration: Duration of the simulation in years
            num_arms: Number of spiral arms (2-4 typical for spiral galaxies)
            arm_width: Width of the spiral arms as a fraction of the arm separation
            arm_tightness: Controls how tightly wound the spiral arms are (0.1-0.3 typical)
            bulge_size: Size of the central bulge as a fraction of galaxy radius
        """
        self.diameter = diameter
        self.radius = diameter / 2
        self.num_stars = num_stars
        self.num_radio_emitting = min(num_radio_emitting, num_stars)
        self.radio_signal_duration = radio_signal_duration
        self.radio_civilization_lifetime = radio_civilization_lifetime
        self.radio_blast_duration = radio_blast_duration
        self.simulation_duration = simulation_duration
        
        # Spiral galaxy parameters
        self.num_arms = num_arms
        self.arm_width = arm_width
        self.arm_tightness = arm_tightness
        self.bulge_size = bulge_size
        
        # Height of the galaxy (saucer/ellipsoid shape)
        self.height = diameter / 20
        
        # Scale factor for density distribution (controls how concentrated stars are toward center)
        self.density_scale = self.radius / 3
        
        # Distribution parameters
        self.arm_fraction = 0.35  # Fraction of stars in spiral arms vs. disk
        self.bulge_fraction = 0.2  # Fraction of stars in the central bulge
        
        # Generate stars
        self.stars = []
        self.earth = None
        self.generate_stars()
    
    def _is_in_spiral_arm(self, r, theta):
        """
        Determine if a point (r, theta) is within a spiral arm.
        
        Args:
            r: Radius from center
            theta: Angle in radians
            
        Returns:
            True if the point is within a spiral arm, False otherwise
        """
        # Normalized radius (0-1)
        r_norm = r / self.radius
        
        # Skip arm check for the bulge region
        if r_norm < self.bulge_size:
            return False
            
        # Calculate the spiral arm angle at this radius
        # Using logarithmic spiral formula: r = a * e^(b * theta)
        # => theta = ln(r/a) / b
        # We use r_norm as r and arm_tightness as b
        # We need to find which arm is closest to this point
        
        # Calculate reference angle for each arm
        arm_angles = []
        for i in range(self.num_arms):
            # Base angle for this arm
            arm_offset = i * (2 * np.pi / self.num_arms)
            
            # Logarithmic spiral formula (reversed to get theta from r)
            # The minus sign makes the arms wind counter-clockwise
            spiral_angle = np.log(r_norm) / self.arm_tightness
            
            # Calculate distance to this arm (angular)
            arm_angle = (theta - arm_offset - spiral_angle) % (2 * np.pi)
            if arm_angle > np.pi:
                arm_angle = 2 * np.pi - arm_angle
            
            arm_angles.append(arm_angle)
        
        # Find the minimum angular distance to any arm
        min_angle = min(arm_angles)
        
        # Calculate the "width" of arms at this radius
        # Arms get narrower as we move outward
        width_factor = self.arm_width * (1 - 0.3 * r_norm)
        
        # Check if the point is within an arm
        return min_angle < (width_factor * np.pi)
    
    def generate_random_position(self):
        """
        Generate a position for a star using spiral galaxy structure.
        
        Returns:
            A numpy array with (x, y, z) coordinates
        """
        # Determine which component this star belongs to: bulge, arm, or disk
        component = random.random()
        
        if component < self.bulge_fraction:
            # BULGE COMPONENT: Central spheroidal bulge with higher density
            
            # Generate spherical coordinates with higher concentration
            # Use inverse transform sampling with power law distribution
            u = random.random()
            r = self.radius * self.bulge_size * pow(u, 0.5)  # More concentration toward center
            
            # Random 3D direction
            phi = random.uniform(0, np.pi)  # Polar angle
            theta = random.uniform(0, 2 * np.pi)  # Azimuthal angle
            
            # Convert to Cartesian
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) * (self.height / self.radius) * 0.5  # Flatten slightly
            
        elif component < (self.bulge_fraction + self.arm_fraction):
            # SPIRAL ARM COMPONENT: Stars along the spiral arms
            
            # Loop until we get a point in an arm
            while True:
                # Generate radius with exponential density distribution
                u = random.random()
                r = -self.density_scale * np.log(1 - u * (1 - np.exp(-self.radius / self.density_scale)))
                
                # Random angle
                theta = random.uniform(0, 2 * np.pi)
                
                # Check if this point is in a spiral arm
                if r <= self.radius and self._is_in_spiral_arm(r, theta):
                    break
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Z coordinate is compressed to create the ellipsoid/saucer shape
            max_height = self.height/2 * (1 - 0.5 * r / self.radius)
            z = random.uniform(-max_height, max_height)
            
        else:
            # DISK COMPONENT: Background stars in the galactic disk
            
            # Use rejection sampling to create exponential density distribution
            while True:
                # Generate angle uniformly
                theta = random.uniform(0, 2 * np.pi)
                
                # Generate radius with exponential density distribution
                u = random.random()
                r = -self.density_scale * np.log(1 - u * (1 - np.exp(-self.radius / self.density_scale)))
                
                # Ensure we're within galaxy radius
                if r <= self.radius:
                    break
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Z coordinate is compressed to create the ellipsoid/saucer shape
            # Scale z compression based on distance from center for a more realistic bulge
            max_height = self.height/2 * (1 - 0.5 * r / self.radius)
            z = random.uniform(-max_height, max_height)
        
        return np.array([x, y, z])
    
    def generate_stars(self):
        """Generate stars within the galaxy."""
        self.stars = []
        
        # Determine which stars will be radio-emitting
        radio_emitting_indices = set(random.sample(range(self.num_stars), self.num_radio_emitting))
        
        # Generate all stars
        for i in range(self.num_stars):
            position = self.generate_random_position()
            is_radio_emitting = i in radio_emitting_indices
            
            # Create a new star
            star = Star(
                position=position,
                is_radio_emitting=is_radio_emitting,
                is_earth=(i == 0),  # First star is Earth
                radio_signal_duration=self.radio_signal_duration,
                radio_civilization_lifetime=self.radio_civilization_lifetime,
                radio_blast_duration=self.radio_blast_duration,
                simulation_duration=self.simulation_duration,
                galaxy_radius=self.radius  # Pass galaxy radius for color calculation
            )
            
            self.stars.append(star)
            
            # Save reference to Earth star
            if i == 0:
                self.earth = star
    
    def get_stars(self):
        """Get all stars in the galaxy."""
        return self.stars
    
    def get_earth(self):
        """Get the Earth star."""
        return self.earth
    
    def get_radio_emitting_stars(self):
        """Get all radio-emitting stars."""
        return [star for star in self.stars if star.is_radio_emitting]