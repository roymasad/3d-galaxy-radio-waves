import numpy as np
import random
from galaxy_model import GalaxyModel
from radio_wave import RadioWave


class Simulation:
    """
    Simulation engine for the radio wave propagation model.
    
    This class manages the simulation state, updates, and detection logic.
    """
    def __init__(self, galaxy_diameter=2000, num_stars=500, num_radio_emitting=300,
                 simulation_speed=1000, simulation_duration=1000000, 
                 radio_signal_duration=10000, radio_civilization_lifetime=100, 
                 radio_blast_duration=10, sky_coverage_percent=0.1):
        """
        Initialize the simulation.
        
        Args:
            galaxy_diameter: Galaxy diameter in light-years
            num_stars: Total number of stars in the galaxy
            num_radio_emitting: Number of stars that emit radio waves
            simulation_speed: Simulation speed multiplier
            simulation_duration: Duration of the simulation in years
            radio_signal_duration: Duration of radio signal emission in years
            radio_civilization_lifetime: How long the civilization emits radio waves
            radio_blast_duration: Duration of a single radio burst in years
            sky_coverage_percent: Percentage of sky covered by radio telescopes
        """
        self.galaxy_diameter = galaxy_diameter
        self.num_stars = num_stars
        self.num_radio_emitting = num_radio_emitting
        self.simulation_speed = simulation_speed
        self.simulation_duration = simulation_duration
        self.radio_signal_duration = radio_signal_duration
        self.radio_civilization_lifetime = radio_civilization_lifetime
        self.radio_blast_duration = radio_blast_duration
        self.sky_coverage_percent = sky_coverage_percent
        
        # Initialize galaxy model
        self.galaxy = GalaxyModel(
            diameter=galaxy_diameter,
            num_stars=num_stars,
            num_radio_emitting=num_radio_emitting,
            radio_signal_duration=radio_signal_duration,
            radio_civilization_lifetime=radio_civilization_lifetime,
            radio_blast_duration=radio_blast_duration,
            simulation_duration=simulation_duration
        )
        
        # Current simulation time in years
        self.current_time = 0
        
        # Active radio waves in the simulation
        self.radio_waves = []
        
        # Tracking hits at Earth
        self.total_hits = 0
        self.hits_log = []  # To track when hits occur
        self.detected_bursts = set()  # Track which bursts have been detected
        
        # Add counter for total bursts emitted
        self.total_bursts_emitted = 0
        
        # Simulation state
        self.is_running = False
        
        # Speed of light in the simulation (light-years per year)
        self.speed_of_light = 1  # 1 light-year per year
    
    def reset(self):
        """Reset the simulation to its initial state."""
        self.__init__(
            galaxy_diameter=self.galaxy_diameter,
            num_stars=self.num_stars,
            num_radio_emitting=self.num_radio_emitting,
            simulation_speed=self.simulation_speed,
            simulation_duration=self.simulation_duration,
            radio_signal_duration=self.radio_signal_duration,
            radio_civilization_lifetime=self.radio_civilization_lifetime,
            radio_blast_duration=self.radio_blast_duration,
            sky_coverage_percent=self.sky_coverage_percent
        )
    
    def start(self):
        """Start or resume the simulation."""
        self.is_running = True
    
    def pause(self):
        """Pause the simulation."""
        self.is_running = False
    
    def is_burst_detected(self, star_id, burst_id):
        """
        Determine if a burst is detected based on sky coverage percentage.
        This function will return the same result for a specific burst every time,
        ensuring consistency when a burst is continuously detected.
        
        Args:
            star_id: ID of the star emitting the burst
            burst_id: ID of the burst
            
        Returns:
            True if the burst is detected, False otherwise
        """
        burst_key = f"{star_id}-{burst_id}"
        
        # If we've already made a determination for this burst, return it
        if burst_key in self.detected_bursts:
            return True
            
        # If sky coverage is 100%, always detect the burst
        if self.sky_coverage_percent >= 100:
            self.detected_bursts.add(burst_key)
            return True
            
        # Otherwise, there's a probability equal to sky_coverage_percent/100
        # that the burst will be detected
        if random.random() <= (self.sky_coverage_percent / 100):
            self.detected_bursts.add(burst_key)
            return True
            
        return False
    
    def update(self, dt):
        """
        Update the simulation state.
        
        Args:
            dt: Time delta in simulation years
        
        Returns:
            True if the simulation is still running, False if it's completed
        """
        if not self.is_running:
            return True
        
        # Scale dt by simulation speed
        scaled_dt = dt * self.simulation_speed
        
        # Update simulation time
        new_time = self.current_time + scaled_dt
        
        # Check for new radio wave emissions
        for star in self.galaxy.get_radio_emitting_stars():
            # Get the burst times for this star during this time step
            burst_times = star.get_burst_times_in_range(self.current_time, new_time)
            
            for burst_time, burst_id in burst_times:
                # Create a new radio wave at the exact burst start time
                wave = RadioWave(
                    source_position=star.position,
                    start_time=burst_time,
                    speed_of_light=self.speed_of_light,
                    galaxy_diameter=self.galaxy_diameter,
                    star_id=star.id,
                    burst_id=burst_id
                )
                self.radio_waves.append(wave)
                self.total_bursts_emitted += 1
        
        # Check for radio wave hits at Earth
        earth = self.galaxy.get_earth()
        # Only count hits if Earth civilization is active during this time step
        earth_civ_active = (earth.is_civilization_active_at(self.current_time) or 
                           earth.is_civilization_active_at(new_time) or 
                           (earth.civilization_start > self.current_time and 
                            earth.civilization_end < new_time))
        
        if earth_civ_active:
            for wave in self.radio_waves:
                # If the wave is now hitting Earth
                if (not wave.is_intersecting_star(earth.position, self.current_time) and
                   wave.is_intersecting_star(earth.position, new_time)):
                    # Check if this burst should be detected based on sky coverage
                    if self.is_burst_detected(wave.star_id, wave.burst_id):
                        self.total_hits += 1
                        self.hits_log.append(new_time)
        
        # Keep waves only until they reach 30% of galaxy diameter
        self.radio_waves = [wave for wave in self.radio_waves 
                          if wave.get_radius(new_time) <= self.galaxy_diameter * 0.2]
        
        # Update current time
        self.current_time = new_time
        
        # Check if simulation is complete
        if self.current_time >= self.simulation_duration:
            self.is_running = False
            return False
        
        return True
    
    def is_earth_civilization_active(self):
        """
        Check if Earth's civilization is active at the current time.
        
        Returns:
            True if Earth's civilization is active, False otherwise
        """
        earth = self.galaxy.get_earth()
        return earth.is_civilization_active_at(self.current_time)
    
    def get_active_waves(self):
        """
        Get currently active radio waves for rendering.
        
        Returns:
            A list of tuples (position, radius, color) for each active wave
        """
        active_waves = []
        for wave in self.radio_waves:
            radius = wave.get_radius(self.current_time)
            if radius <= self.galaxy_diameter:
                active_waves.append({
                    'position': wave.source_position,
                    'radius': radius,
                    'color': wave.get_color(self.current_time)
                })
        return active_waves
    
    def get_stars_data(self):
        """
        Get star data for rendering.
        
        Returns:
            A list of dictionaries with star data
        """
        stars_data = []
        for star in self.galaxy.get_stars():
            # Include information about Earth's civilization status
            is_civ_active = star.is_earth and star.is_civilization_active_at(self.current_time)
            
            stars_data.append({
                'position': star.position, 
                'color': star.color, 
                'is_earth': star.is_earth,
                'is_civilization_active': is_civ_active
            })
        
        return stars_data
    
    def get_summary(self):
        """
        Get a summary of the simulation results.
        
        Returns:
            A dictionary with summary data
        """
        return {
            'total_hits': self.total_hits,
            
            'total_time': self.current_time,
            'hits_log': self.hits_log,
            'earth_civilization_active': self.is_earth_civilization_active(),
            'earth_civilization_start': self.galaxy.get_earth().civilization_start,
            'earth_civilization_end': self.galaxy.get_earth().civilization_end,
            'sky_coverage': self.sky_coverage_percent,
            'total_bursts_emitted': self.total_bursts_emitted
        }