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
        
        # Edge softening parameters
        self.edge_softness = 0.15  # How much beyond nominal radius stars can appear (as fraction of radius)
        self.edge_density_falloff = 2.5  # Higher values create sharper density falloff at edges
        
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
    
    def _calculate_edge_probability(self, r):
        """
        Calculate the probability of accepting a star based on its distance from center.
        Creates a soft edge instead of a hard cutoff at galaxy radius.
        
        Args:
            r: Radius from center
            
        Returns:
            Probability (0-1) of accepting this star
        """
        # Stars inside the nominal radius are always accepted
        if r <= self.radius:
            return 1.0
        
        # Stars beyond the extended radius are never accepted
        extended_radius = self.radius * (1.0 + self.edge_softness)
        if r >= extended_radius:
            return 0.0
        
        # For stars in the soft edge region, probability decreases with distance
        edge_fraction = (r - self.radius) / (extended_radius - self.radius)
        return 1.0 - pow(edge_fraction, self.edge_density_falloff)
    
    def _calculate_z_height(self, r):
        """
        Calculate the maximum height (z-coordinate) for a star at radius r.
        Creates a tapered disk that's thinner at the edges.
        
        Args:
            r: Radius from center
            
        Returns:
            Maximum height value
        """
        # Normalize radius
        r_norm = min(1.0, r / self.radius)
        
        # Use a quadratic falloff for more rapid thinning near the edges
        # This creates a smoother tapering effect
        height_factor = 1.0 - (r_norm ** 1.8)
        
        # Further reduce thickness at the very edge
        if r_norm > 0.8:
            edge_factor = (1.0 - r_norm) / 0.2  # Goes from 1 to 0 as r_norm goes from 0.8 to 1.0
            height_factor *= edge_factor
        
        return self.height/2 * height_factor
    
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
            
            # Loop until we get a point in an arm with accepted edge probability
            while True:
                # Generate radius with exponential density distribution
                u = random.random()
                r = -self.density_scale * np.log(1 - u * (1 - np.exp(-self.radius / self.density_scale)))
                
                # Apply soft edge - possibly extend beyond nominal radius
                extended_radius = self.radius * (1.0 + self.edge_softness)
                if r > extended_radius:
                    continue
                
                # Apply edge probability
                edge_prob = self._calculate_edge_probability(r)
                if random.random() > edge_prob:
                    continue
                
                # Random angle
                theta = random.uniform(0, 2 * np.pi)
                
                # Check if this point is in a spiral arm
                # Allow stars to be outside arms more often near the edge
                r_norm = min(1.0, r / self.radius)
                arm_check_prob = 1.0 - 0.5 * max(0, r_norm - 0.9) / 0.1  # Reduce arm constraint at edges
                
                if self._is_in_spiral_arm(r, theta) or (r_norm > 0.9 and random.random() > arm_check_prob):
                    break
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Z coordinate with tapering at edges
            max_height = self._calculate_z_height(r)
            z = random.uniform(-max_height, max_height)
            
        else:
            # DISK COMPONENT: Background stars in the galactic disk
            
            # Use rejection sampling with soft edge
            while True:
                # Generate angle uniformly
                theta = random.uniform(0, 2 * np.pi)
                
                # Generate radius with exponential density distribution
                u = random.random()
                r = -self.density_scale * np.log(1 - u * (1 - np.exp(-self.radius / self.density_scale)))
                
                # Apply soft edge
                extended_radius = self.radius * (1.0 + self.edge_softness)
                if r > extended_radius:
                    continue
                    
                # Apply edge probability
                edge_prob = self._calculate_edge_probability(r)
                if random.random() <= edge_prob:
                    break
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Z coordinate with tapering at edges
            max_height = self._calculate_z_height(r)
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