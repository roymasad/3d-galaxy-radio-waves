import numpy as np


class RadioWave:
    """
    Represents a spherical radio wave propagating through space.
    """
    def __init__(self, source_position, start_time, speed_of_light=1, galaxy_diameter=2000,
                 star_id=None, burst_id=None):
        """
        Initialize a radio wave.
        
        Args:
            source_position: Origin point of the wave as [x, y, z]
            start_time: Time when the wave starts propagating
            speed_of_light: Speed of light in simulation units (light-years/year)
            galaxy_diameter: Diameter of the galaxy for fade calculation
            star_id: ID of the star that emitted the wave
            burst_id: ID of the specific burst from the star
        """
        self.source_position = source_position
        self.start_time = start_time
        self.speed_of_light = speed_of_light
        self.galaxy_diameter = galaxy_diameter
        self.base_color = (0.0, 1.0,0.0)  # green color for better visibility
        self.star_id = star_id
        self.burst_id = burst_id
    
    def get_radius(self, current_time):
        """
        Get the radius of the wave at the given time.
        
        Args:
            current_time: Current simulation time in years
            
        Returns:
            Radius of the wave in light-years
        """
        if current_time < self.start_time:
            return 0
            
        return (current_time - self.start_time) * self.speed_of_light
    
    def get_color(self, current_time):
        """
        Get the color of the wave at the given time, including opacity.
        
        Args:
            current_time: Current simulation time in years
            
        Returns:
            RGBA color tuple with calculated opacity
        """
        radius = self.get_radius(current_time)
        max_radius = self.galaxy_diameter * 0.2  # Maximum radius before removal
        
        # Calculate opacity based on radius
        # Start at 50% opacity and fade to 0% at max_radius
        opacity = 0.5 - (radius / max_radius)
        opacity = max(0.0, min(0.5, opacity))  # Clamp between 0 and 0.5
        
        return (*self.base_color, opacity)
        
    def is_intersecting_star(self, star_position, time):
        """
        Check if the wave has reached a given star at the specified time.
        
        Args:
            star_position: Position of the star to check [x, y, z]
            time: Time at which to check for intersection
            
        Returns:
            True if the wave has reached the star, False otherwise
        """
        if time < self.start_time:
            return False
            
        # Get wave radius at the current time
        radius = self.get_radius(time)
        
        # Calculate distance from wave source to star
        distance = sum((a - b) ** 2 for a, b in zip(self.source_position, star_position)) ** 0.5
        
        # The wave intersects if the distance is less than or equal to the radius
        return distance <= radius