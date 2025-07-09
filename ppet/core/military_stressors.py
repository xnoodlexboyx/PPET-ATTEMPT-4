"""Military-grade environmental stressor models for PUF simulation.

This module provides comprehensive models for simulating military-specific
environmental conditions and their effects on PUF behavior.

References:
    - MIL-STD-810H: Environmental Engineering Considerations
    - MIL-STD-461G: Electromagnetic Interference
    - MIL-STD-883K: Microcircuit Test Methods
"""

from typing import Dict, Optional, Union, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

class MilitaryEnvironment(Enum):
    """Military environment profiles based on MIL-STD-810H."""
    GROUND_MOBILE = "ground_mobile"
    AIRCRAFT_INTERNAL = "aircraft_internal"
    AIRCRAFT_EXTERNAL = "aircraft_external"
    NAVAL_SHELTERED = "naval_sheltered"
    NAVAL_EXPOSED = "naval_exposed"
    SPACE_VEHICLE = "space_vehicle"

@dataclass
class TemperatureProfile:
    """Temperature cycling profile parameters."""
    min_temp: float  # °C
    max_temp: float  # °C
    cycle_period: float  # hours
    dwell_time: float  # hours at extreme temperatures
    ramp_rate: float  # °C/minute

@dataclass
class EMIProfile:
    """Electromagnetic interference profile parameters."""
    conducted_susceptibility: float  # V
    radiated_susceptibility: float  # V/m
    frequency_range: tuple  # Hz
    pulse_width: float  # seconds
    repetition_rate: float  # Hz

class MilitaryStressors:
    """Military-grade environmental stressor simulator."""
    
    def __init__(
        self,
        environment: MilitaryEnvironment,
        mission_duration: float = 1000.0,  # hours
        seed: Optional[int] = None
    ):
        """Initialize military stressor simulator.
        
        Args:
            environment: Military environment profile
            mission_duration: Total mission duration in hours
            seed: Random seed for reproducibility
        """
        self.environment = environment
        self.mission_duration = mission_duration
        if seed is not None:
            np.random.seed(seed)
            
        self._initialize_profiles()
        
    def _initialize_profiles(self):
        """Initialize environment-specific profiles."""
        # Temperature profiles based on MIL-STD-810H
        self.temp_profiles = {
            MilitaryEnvironment.GROUND_MOBILE: TemperatureProfile(
                min_temp=-40.0, max_temp=85.0,
                cycle_period=24.0, dwell_time=4.0, ramp_rate=5.0
            ),
            MilitaryEnvironment.AIRCRAFT_INTERNAL: TemperatureProfile(
                min_temp=-45.0, max_temp=70.0,
                cycle_period=12.0, dwell_time=2.0, ramp_rate=10.0
            ),
            MilitaryEnvironment.AIRCRAFT_EXTERNAL: TemperatureProfile(
                min_temp=-55.0, max_temp=125.0,
                cycle_period=6.0, dwell_time=1.0, ramp_rate=15.0
            ),
            MilitaryEnvironment.NAVAL_SHELTERED: TemperatureProfile(
                min_temp=-10.0, max_temp=65.0,
                cycle_period=24.0, dwell_time=6.0, ramp_rate=2.0
            ),
            MilitaryEnvironment.NAVAL_EXPOSED: TemperatureProfile(
                min_temp=-25.0, max_temp=55.0,
                cycle_period=12.0, dwell_time=4.0, ramp_rate=3.0
            ),
            MilitaryEnvironment.SPACE_VEHICLE: TemperatureProfile(
                min_temp=-65.0, max_temp=125.0,
                cycle_period=1.5, dwell_time=0.25, ramp_rate=20.0
            )
        }
        
        # EMI profiles based on MIL-STD-461G
        self.emi_profiles = {
            MilitaryEnvironment.GROUND_MOBILE: EMIProfile(
                conducted_susceptibility=10.0,
                radiated_susceptibility=200.0,
                frequency_range=(10e3, 18e9),
                pulse_width=1e-6,
                repetition_rate=1000.0
            ),
            # ... similar profiles for other environments ...
        }
        
        self.current_profile = self.temp_profiles[self.environment]
        self.current_emi = self.emi_profiles[self.environment]
        
    def get_temperature_stress(self, time: float) -> float:
        """Calculate temperature at given mission time.
        
        Args:
            time: Mission time in hours
            
        Returns:
            Temperature in °C
        """
        profile = self.current_profile
        cycle_position = (time % profile.cycle_period) / profile.cycle_period
        
        if cycle_position < profile.dwell_time / profile.cycle_period:
            # Cold dwell
            return profile.min_temp
        elif cycle_position > (1.0 - profile.dwell_time / profile.cycle_period):
            # Hot dwell
            return profile.max_temp
        else:
            # Temperature ramp
            ramp_position = (cycle_position - profile.dwell_time / profile.cycle_period) / (
                1.0 - 2 * profile.dwell_time / profile.cycle_period)
            return profile.min_temp + ramp_position * (profile.max_temp - profile.min_temp)
    
    def get_emi_stress(self, time: float) -> Dict[str, float]:
        """Calculate EMI stressors at given mission time.
        
        Args:
            time: Mission time in hours
            
        Returns:
            Dictionary of EMI parameters
        """
        profile = self.current_emi
        
        # Base EMI level
        base_emi = np.random.normal(0.5, 0.1)  # Normalized base EMI
        
        # Add periodic interference
        periodic_component = 0.2 * np.sin(2 * np.pi * time / 24.0)  # Daily variation
        
        # Add random pulses
        if np.random.random() < 0.01:  # 1% chance of pulse
            pulse_amplitude = np.random.uniform(0.5, 1.0)
        else:
            pulse_amplitude = 0.0
            
        total_emi = base_emi + periodic_component + pulse_amplitude
        
        return {
            'conducted': total_emi * profile.conducted_susceptibility,
            'radiated': total_emi * profile.radiated_susceptibility,
            'frequency': np.random.uniform(*profile.frequency_range),
            'normalized': total_emi
        }
    
    def get_aging_factor(self, time: float) -> float:
        """Calculate aging degradation factor.
        
        Args:
            time: Mission time in hours
            
        Returns:
            Aging factor (1.0 = no aging, increases with time)
        """
        # Accelerated aging based on temperature cycling
        temp = self.get_temperature_stress(time)
        
        # Arrhenius acceleration factor
        k = 8.617333262145e-5  # Boltzmann constant
        Ea = 0.7  # Activation energy (eV)
        T_use = 273.15 + 25.0  # Reference temp (K)
        T_stress = 273.15 + temp  # Stress temp (K)
        
        acceleration = np.exp((Ea/k) * (1/T_use - 1/T_stress))
        
        # Basic aging model
        base_aging = 1.0 + 0.1 * (time / self.mission_duration)
        
        return base_aging * acceleration
    
    def get_all_stressors(self, time: float) -> Dict[str, float]:
        """Get all environmental stressors at given time.
        
        Args:
            time: Mission time in hours
            
        Returns:
            Dictionary of all stressor values
        """
        emi = self.get_emi_stress(time)
        return {
            'temperature': self.get_temperature_stress(time),
            'aging_factor': self.get_aging_factor(time),
            'em_noise': emi['normalized'],
            'conducted_emi': emi['conducted'],
            'radiated_emi': emi['radiated'],
            'emi_frequency': emi['frequency']
        } 