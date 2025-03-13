import random
import numpy as np


class Star:
    """
    Represents a star in the galaxy simulation.
    
    Each star has a position and may emit radio waves.
    """
    _next_id = 0  # Class variable to generate unique IDs
    
    def __init__(self, position, is_radio_emitting=False, is_earth=False, 
                 radio_signal_duration=10_000, radio_civilization_lifetime=100, 
                 radio_blast_duration=10, simulation_duration=None, galaxy_radius=None):
        """
        Initialize a star.
        
        Args:
            position: 3D coordinates (x, y, z) as numpy array
            is_radio_emitting: Whether the star emits radio waves
            is_earth: Whether this star represents Earth
            radio_signal_duration: Duration of radio signal emission in years
            radio_civilization_lifetime: How long the civilization emits radio waves
            radio_blast_duration: Duration of a single radio burst in years
            simulation_duration: Duration of the simulation in years
            galaxy_radius: The radius of the galaxy for color calculations
        """
        self.id = Star._next_id
        Star._next_id += 1
        
        self.position = position
        self.is_radio_emitting = is_radio_emitting
        self.is_earth = is_earth
        self.radio_signal_duration = radio_signal_duration
        self.radio_blast_duration = radio_blast_duration
        self.galaxy_radius = galaxy_radius
        
        # If simulation duration wasn't provided, use a default value
        if simulation_duration is None:
            simulation_duration = radio_signal_duration * 10
            
        self.simulation_duration = simulation_duration
        
        # When this star starts emitting radio waves (if it does)
        if is_radio_emitting:
            # Ensure radio emissions occur within the simulation timeframe
            # Leave room at the end for the full signal duration
            latest_possible_start = max(0, simulation_duration - radio_signal_duration)
            
            # Randomly pick a start time within the simulation duration
            self.radio_emission_start = random.uniform(0, latest_possible_start)
            self.radio_emission_end = self.radio_emission_start + radio_signal_duration
            
            # Generate burst times within the emission period
            self.bursts = self._generate_burst_schedule()
        else:
            self.radio_emission_start = None
            self.radio_emission_end = None
            self.bursts = []
        
        # Earth civilization properties
        self.radio_civilization_lifetime = radio_civilization_lifetime
        if is_earth:
            # For Earth, calculate when its civilization becomes capable of detecting radio signals
            # Ensure the civilization period fits within the simulation timeframe
            latest_possible_start = max(0, simulation_duration - radio_civilization_lifetime)
            self.civilization_start = random.uniform(0, latest_possible_start)
            self.civilization_end = self.civilization_start + radio_civilization_lifetime
        else:
            self.civilization_start = None
            self.civilization_end = None
        
        # Color for visualization - radio emitting stars and Earth have special colors
        if is_earth:
            self.color = (0, 0, 1, 1)  # Blue for Earth
        elif is_radio_emitting:
            self.color = (1, 0, 0, 1)  # Red for radio-emitting stars
        else:
            # Calculate color based on distance from galaxy center for normal stars
            self.color = self._calculate_star_color()
    
    def _calculate_star_color(self):
        """
        Calculate star color based on distance from galaxy center.
        Closer stars are whiter, distant stars are more gray.
        
        Returns:
            RGBA color tuple
        """
        # Default white if no galaxy radius provided
        if self.galaxy_radius is None:
            return (1, 1, 1, 1)
            
        # Calculate distance from galaxy center
        distance = np.linalg.norm(self.position)
        
        # Normalize distance (0 at center, 1 at galaxy radius)
        normalized_distance = min(1.0, distance / self.galaxy_radius)
        
        # Linear interpolation between white (1,1,1) and gray (0.5,0.5,0.5)
        # The closer to center (normalized_distance->0), the whiter the star
        # The further from center (normalized_distance->1), the grayer the star
        color_value = 1.0 - (0.9 * normalized_distance)
        
        # 10 percent change to convert color from white to yellow
        color_value2 = color_value
        if random.random() < 0.1:
            color_value2 = 0
        
        return (color_value, color_value, color_value2, 1)
    
    def _generate_burst_schedule(self):
        """
        Generate a schedule of radio bursts during the civilization's emission period.
        
        Returns:
            A list of tuples (start_time, burst_id)
        """
        bursts = []
        
        # If civilization doesn't have a radio emission period, return empty list
        if not self.is_radio_emitting or self.radio_emission_start is None:
            return bursts
        
        # Calculate emission period
        emission_duration = self.radio_emission_end - self.radio_emission_start
        
        # If burst duration is 0 or larger than the emission period, just use one burst
        # for the entire emission period
        if self.radio_blast_duration <= 0 or self.radio_blast_duration >= emission_duration:
            bursts.append((self.radio_emission_start, 0))
            return bursts
        
        # Generate bursts throughout the emission period
        burst_id = 0
        current_time = self.radio_emission_start
        
        while current_time < self.radio_emission_end:
            bursts.append((current_time, burst_id))
            burst_id += 1
            
            # Add the burst duration to get the next burst start time
            current_time += self.radio_blast_duration
        
        return bursts
    
    def get_burst_times_in_range(self, start_time, end_time):
        """
        Get all burst start times that occur within a specific time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of (burst_time, burst_id) tuples for bursts that start within the range
        """
        return [(time, burst_id) for time, burst_id in self.bursts 
                if start_time < time <= end_time]
    
    def is_emitting_at(self, time):
        """
        Check if the star is emitting radio waves at the given time.
        
        Args:
            time: The current simulation time in years
            
        Returns:
            True if the star is emitting at the given time, False otherwise
        """
        if not self.is_radio_emitting:
            return False
        
        # Check if the time falls within any burst period
        for burst_start, burst_id in self.bursts:
            burst_end = burst_start + self.radio_blast_duration
            if burst_start <= time <= burst_end:
                return True
                
        return False
        
    def is_civilization_active_at(self, time):
        """
        Check if the star's civilization is active at the given time.
        
        Args:
            time: The current simulation time in years
            
        Returns:
            True if the civilization is active at the given time, False otherwise
        """
        if not self.is_earth:
            return False
        
        return self.civilization_start <= time <= self.civilization_end