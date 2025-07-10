"""Harsh environment and high-security environmental stressor models for PUF simulation.

This module provides comprehensive models for simulating extreme environmental
conditions and their effects on PUF behavior in harsh deployment scenarios.

References:
    - Military approximations for environmental engineering considerations
    - Military approximations for electromagnetic interference
    - Military approximations for microcircuit test methods
"""

from typing import Dict, Optional, Union, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

class MilitaryEnvironment(Enum):
    """Harsh environment profiles based on military approximations."""
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
    """Harsh environment and high-security environmental stressor simulator."""
    
    def __init__(
        self,
        environment: MilitaryEnvironment,
        mission_duration: float = 1000.0,  # hours
        seed: Optional[int] = None
    ):
        """Initialize environmental stressor simulator.
        
        Args:
            environment: Environmental profile
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
        # Temperature profiles based on military approximation
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
        
        # EMI profiles based on military approximation
        self.emi_profiles = {
            MilitaryEnvironment.GROUND_MOBILE: EMIProfile(
                conducted_susceptibility=10.0,
                radiated_susceptibility=200.0,
                frequency_range=(10e3, 18e9),
                pulse_width=1e-6,
                repetition_rate=1000.0
            ),
            MilitaryEnvironment.AIRCRAFT_INTERNAL: EMIProfile(
                conducted_susceptibility=5.0,
                radiated_susceptibility=50.0,
                frequency_range=(10e3, 40e9),
                pulse_width=5e-7,
                repetition_rate=2000.0
            ),
            MilitaryEnvironment.AIRCRAFT_EXTERNAL: EMIProfile(
                conducted_susceptibility=15.0,
                radiated_susceptibility=500.0,
                frequency_range=(10e3, 40e9),
                pulse_width=2e-6,
                repetition_rate=1500.0
            ),
            MilitaryEnvironment.NAVAL_SHELTERED: EMIProfile(
                conducted_susceptibility=8.0,
                radiated_susceptibility=100.0,
                frequency_range=(10e3, 18e9),
                pulse_width=1.5e-6,
                repetition_rate=800.0
            ),
            MilitaryEnvironment.NAVAL_EXPOSED: EMIProfile(
                conducted_susceptibility=12.0,
                radiated_susceptibility=300.0,
                frequency_range=(10e3, 18e9),
                pulse_width=2e-6,
                repetition_rate=1200.0
            ),
            MilitaryEnvironment.SPACE_VEHICLE: EMIProfile(
                conducted_susceptibility=20.0,
                radiated_susceptibility=1000.0,
                frequency_range=(10e3, 100e9),
                pulse_width=1e-7,
                repetition_rate=5000.0
            )
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
            
        total_emi = base_emi + abs(periodic_component) + pulse_amplitude
        # Ensure non-negativity at the source
        total_emi = max(0, total_emi)
        
        return {
            'conducted': total_emi * profile.conducted_susceptibility,
            'radiated': total_emi * profile.radiated_susceptibility,
            'frequency': np.random.uniform(*profile.frequency_range),
            'normalized': total_emi
        }
    
    def get_aging_factor(self, time: float) -> float:
        """
        Calculate a monotonic and cumulative aging degradation factor.
        """
        if time <= 0:
            return 1.0

        # Use a more stable integration approach
        time_steps = np.linspace(0, time, int(time * 4) + 2) # Increased sampling
        temps = np.array([self.get_temperature_stress(t) for t in time_steps])

        # Arrhenius model for temperature-dependent degradation
        k = 8.617333e-5  # Boltzmann constant (eV/K)
        Ea = 0.6  # Activation energy for silicon device aging (eV)
        T_ref = 273.15 + 50.0  # Reference temperature (K)
        T_stress = 273.15 + temps

        # Calculate acceleration factor relative to reference temperature
        acceleration_factors = np.exp((Ea / k) * (1 / T_ref - 1 / T_stress))
        
        # Integrate acceleration over time to get cumulative stress
        cumulative_stress = np.trapz(acceleration_factors, time_steps)

        # Map cumulative stress to an aging factor using proper time constant
        # τ = 8760 hours (1 year) as per documentation
        tau_aging = 8760.0  # hours, from parameter_validation.md
        alpha_aging = 0.1   # maximum aging factor from documentation
        
        # Use exponential aging model: E_aging(t) = α_aging × (1 - exp(-t/τ))
        base_aging = alpha_aging * (1.0 - np.exp(-time / tau_aging))
        
        # Add temperature-dependent acceleration
        temp_acceleration = cumulative_stress / (time * len(time_steps)) if time > 0 else 1.0
        aging_factor = 1.0 + base_aging * temp_acceleration
        
        return aging_factor
    
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