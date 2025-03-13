import numpy as np
import random
from star import Star


class GalaxyModel:
    """
    Represents a 3D galaxy model with stars.
    
    The galaxy is modeled as an ellipsoid (saucer shape) with higher density towards the center.
    """
    def __init__(self, diameter=2000, num_stars=500, num_radio_emitting=300,
                 radio_signal_duration=10_000, radio_civilization_lifetime=100, 
                 radio_blast_duration=10, simulation_duration=1_000_000):
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
        """
        self.diameter = diameter
        self.radius = diameter / 2
        self.num_stars = num_stars
        self.num_radio_emitting = min(num_radio_emitting, num_stars)
        self.radio_signal_duration = radio_signal_duration
        self.radio_civilization_lifetime = radio_civilization_lifetime
        self.radio_blast_duration = radio_blast_duration
        self.simulation_duration = simulation_duration
        
        # Height of the galaxy (saucer/ellipsoid shape)
        self.height = diameter / 20
        
        # Scale factor for density distribution (controls how concentrated stars are toward center)
        self.density_scale = self.radius / 3
        
        # Generate stars
        self.stars = []
        self.earth = None
        self.generate_stars()
    
    def generate_random_position(self):
        """
        Generate a position within the galaxy ellipsoid with higher density toward the center.
        Uses an exponential density profile typical of disk galaxies.
        
        Returns:
            A numpy array with (x, y, z) coordinates
        """
        # Use rejection sampling to create exponential density distribution
        while True:
            # Generate angle uniformly
            theta = random.uniform(0, 2 * np.pi)
            
            # Generate radius with exponential density distribution
            # Using inverse transform sampling
            u = random.random()
            # Use exponential distribution with scale factor
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