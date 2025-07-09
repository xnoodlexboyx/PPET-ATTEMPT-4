import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .military_stressors import MilitaryStressors, MilitaryEnvironment

class PUF:
    def __init__(
        self,
        seed: Optional[int] = None,
        environmental_stressors: Optional[Dict[str, float]] = None,
        military_environment: Optional[MilitaryEnvironment] = None,
        mission_time: float = 0.0
    ):
        """Initialize base PUF class.
        
        Args:
            seed: Random seed for reproducibility
            environmental_stressors: Dictionary of environmental conditions
                Example: {'temperature': 25.0, 'voltage': 1.2, 'em_noise': 0.0}
            military_environment: Optional military environment profile
            mission_time: Current mission time in hours (for military environments)
        """
        self.seed = seed
        self.mission_time = mission_time
        
        # Initialize military stressor simulator if environment specified
        self.military_stressors = None
        if military_environment is not None:
            self.military_stressors = MilitaryStressors(
                environment=military_environment,
                mission_duration=1000.0,  # Default 1000 hour mission
                seed=seed
            )
            # Get initial stressor values
            mil_stressors = self.military_stressors.get_all_stressors(mission_time)
            self.environmental_stressors = {
                'temperature': mil_stressors['temperature'],
                'voltage': 1.2,  # Nominal voltage
                'em_noise': mil_stressors['em_noise'],
                'aging_factor': mil_stressors['aging_factor']
            }
        else:
            self.environmental_stressors = environmental_stressors or {
                'temperature': 25.0,  # °C
                'voltage': 1.2,      # V
                'em_noise': 0.0,     # normalized units
                'aging_factor': 1.0  # no aging
            }
            
        if seed is not None:
            np.random.seed(seed)

    def update_mission_time(self, time: float):
        """Update mission time and environmental stressors.
        
        Args:
            time: New mission time in hours
        """
        self.mission_time = time
        if self.military_stressors is not None:
            mil_stressors = self.military_stressors.get_all_stressors(time)
            self.environmental_stressors.update({
                'temperature': mil_stressors['temperature'],
                'em_noise': mil_stressors['em_noise'],
                'aging_factor': mil_stressors['aging_factor']
            })

    def _apply_environmental_effects(self, value: float) -> float:
        """Apply environmental stressor effects to a value.
        
        Based on:
        - Temperature: Linear effect (±0.1% per °C from 25°C)
        - Voltage: Quadratic effect around nominal
        - EM noise: Additive Gaussian noise
        - Aging: Multiplicative degradation
        """
        # Temperature effect with wider range support
        temp_effect = 1.0 + 0.001 * (self.environmental_stressors['temperature'] - 25.0)
        
        # Voltage effect
        voltage_nominal = 1.2
        voltage_effect = 1.0 + 0.05 * ((self.environmental_stressors['voltage'] - voltage_nominal) / voltage_nominal) ** 2
        
        # Enhanced EM noise model
        em_noise = np.random.normal(0, self.environmental_stressors['em_noise'] * 0.05)
        
        # Apply aging degradation
        aging_effect = self.environmental_stressors.get('aging_factor', 1.0)
        
        # Combine all effects
        return (value * temp_effect * voltage_effect * aging_effect) + em_noise

    def generate_crps(self, num_crps: int) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Generate challenge-response pairs."""
        raise NotImplementedError

class ArbiterPUF(PUF):
    def __init__(
        self,
        challenge_length: Optional[int] = None,
        n_stages: Optional[int] = None,
        seed: Optional[int] = None,
        environmental_stressors: Optional[Dict[str, float]] = None
    ):
        """Initialize Arbiter PUF.
        
        Args:
            challenge_length: Number of challenge bits (preferred parameter name)
            n_stages: Number of stages in the arbiter chain (legacy parameter name)
            seed: Random seed for reproducibility
            environmental_stressors: Environmental conditions
        
        Note:
            Either challenge_length or n_stages must be provided. 
            challenge_length takes precedence if both are given.
        
        References:
            - Gassend et al. "Silicon Physical Random Functions" (CCS 2002)
            - Majzoobi et al. "Testing Techniques for Hardware Security" (ITC 2008)
        """
        super().__init__(seed, environmental_stressors)
        
        # Handle parameter compatibility
        if challenge_length is not None:
            self.n_stages = challenge_length
        elif n_stages is not None:
            self.n_stages = n_stages
        else:
            raise ValueError("Either challenge_length or n_stages must be provided")
        
        # Store both for backward compatibility
        self.challenge_length = self.n_stages
        
        # Initialize delay parameters based on manufacturing variations
        # Using hierarchical variation model: global + local variations
        self.global_variation = np.random.normal(0, 0.4)  # 40% global variation
        
        # Generate systematic variations (position-dependent)
        position = np.linspace(-1, 1, self.n_stages)
        gradient = np.random.normal(0, 0.2)  # 20% gradient variation
        self.systematic_variation = gradient * position  # Linear gradient across stages
        
        # Generate local variations for each stage and path
        # Increased variation and ensured independence between paths
        self.local_variations_top = np.random.normal(0, 0.3, size=self.n_stages)  # 30% local variation
        self.local_variations_bottom = np.random.normal(0, 0.3, size=self.n_stages)  # 30% local variation
        
        # Generate arbiter bias
        self.arbiter_bias = np.random.normal(0, 0.15)  # 15% arbiter bias
        
        # Generate stage-specific environmental sensitivities
        # Ensure some stages are more sensitive than others
        base_sensitivities = np.random.normal(1.0, 0.2, size=self.n_stages)  # ±20% variation in base sensitivity
        self.temp_sensitivities = base_sensitivities * np.random.normal(1.0, 0.1, size=self.n_stages)
        self.voltage_sensitivities = base_sensitivities * np.random.normal(1.0, 0.1, size=self.n_stages)
        self.noise_sensitivities = base_sensitivities * np.random.normal(1.0, 0.1, size=self.n_stages)

    def _apply_stage_environmental_effects(self, stage_idx: int, value: float) -> float:
        """Apply environmental effects to a specific stage.
        
        Different stages have slightly different sensitivities to environmental conditions.
        This models the physical reality that manufacturing variations affect not just
        the delays but also how sensitive each stage is to environmental conditions.
        """
        # Base environmental effects (further reduced sensitivity)
        temp_effect = 1.0 + 0.00005 * (self.environmental_stressors['temperature'] - 25.0)  # Reduced from 0.0001
        voltage_nominal = 1.2
        voltage_ratio = self.environmental_stressors['voltage'] / voltage_nominal
        voltage_effect = 1.0 + 0.002 * ((voltage_ratio - 1) ** 2)  # Reduced from 0.005
        
        # Apply stage-specific sensitivities
        temp_effect = temp_effect ** self.temp_sensitivities[stage_idx]
        voltage_effect = voltage_effect ** self.voltage_sensitivities[stage_idx]
        
        # Apply effects
        value *= temp_effect * voltage_effect
        
        # Add noise with stage-specific sensitivity
        noise_amplitude = 0.002 * self.noise_sensitivities[stage_idx]  # Reduced from 0.005
        noise = np.random.normal(0, self.environmental_stressors['em_noise'] * noise_amplitude)
        
        return value + noise

    def evaluate(self, challenge: np.ndarray) -> int:
        """Evaluate PUF response for a given challenge.
        
        Implements accurate delay accumulation model with:
        - Path switching based on challenge bits
        - Accumulated delay differences
        - Environmental effects
        - Non-linear path interactions
        """
        assert len(challenge) == self.n_stages
        
        # Initialize delay difference
        delay_diff = 0.0
        
        # Track current path state
        path_state = 0.0  # Represents accumulated switching effect
        
        # Accumulate delays through stages with switching
        for i in range(self.n_stages):
            # Calculate base delays for both paths
            base_delay_top = 1.0 + self.global_variation + self.systematic_variation[i] + self.local_variations_top[i]
            base_delay_bottom = 1.0 + self.global_variation - self.systematic_variation[i] + self.local_variations_bottom[i]
            
            # Apply environmental effects to both paths
            delay_top = self._apply_stage_environmental_effects(i, base_delay_top)
            delay_bottom = self._apply_stage_environmental_effects(i, base_delay_bottom)
            
            # Calculate effective delays with non-linear path interactions
            if challenge[i] == 0:
                # Straight path
                effective_delay = delay_top - delay_bottom
                path_state *= 0.9  # Decay previous switching effects
            else:
                # Crossed path
                effective_delay = delay_bottom - delay_top
                path_state = path_state * 0.9 + 0.1  # Accumulate switching effect
            
            # Add non-linear switching effect
            switching_impact = 0.05 * path_state * (delay_top + delay_bottom) / 2
            effective_delay += switching_impact
            
            # Update delay difference
            delay_diff += effective_delay
        
        # Add arbiter bias with environmental effects
        final_bias = self._apply_stage_environmental_effects(self.n_stages - 1, self.arbiter_bias)
        delay_diff += final_bias
        
        return 1 if delay_diff > 0 else 0

    def generate_crps(
        self,
        num_crps: int,
        challenges: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate challenge-response pairs.
        
        Args:
            num_crps: Number of CRPs to generate
            challenges: Optional predefined challenges
        """
        if challenges is None:
            challenges = np.random.randint(0, 2, size=(num_crps, self.n_stages))
        
        responses = np.array([self.evaluate(ch) for ch in challenges])
        return challenges, responses

    def generate_responses(self, challenges: np.ndarray) -> np.ndarray:
        """Generate responses for given challenges.
        
        Args:
            challenges: Challenge bit vectors
        
        Returns:
            Response bits
        """
        _, responses = self.generate_crps(len(challenges), challenges)
        return responses

class SRAMPUF(PUF):
    def __init__(
        self,
        rows: int,
        columns: int,
        seed: Optional[int] = None,
        environmental_stressors: Optional[Dict[str, float]] = None
    ):
        """Initialize SRAM PUF.
        
        Args:
            rows: Number of SRAM rows
            columns: Number of SRAM columns
            seed: Random seed for reproducibility
            environmental_stressors: Environmental conditions
            
        References:
            - Holcomb et al. "Initial SRAM State as a Source of Randomness" (CHES 2007)
            - Guajardo et al. "FPGA Intrinsic PUFs and Their Use" (CHES 2007)
        """
        super().__init__(seed, environmental_stressors)
        self.rows = rows
        self.columns = columns
        
        # Model transistor mismatch parameters
        # Reduced mismatch variance for better reliability
        self.vth_mismatch = np.random.normal(0, 0.05, size=(rows, columns, 2))  # Reduced from 0.1
        self.beta_mismatch = np.random.normal(1, 0.025, size=(rows, columns, 2))  # Reduced from 0.05

    def generate_startup_state(self) -> np.ndarray:
        """Generate SRAM startup state based on transistor characteristics.
        
        Models:
        - Threshold voltage mismatch
        - Current factor mismatch
        - Temperature and voltage dependence
        - Noise effects
        """
        # Temperature effect on threshold voltage (reduced sensitivity)
        temp_effect = 0.0005 * (self.environmental_stressors['temperature'] - 25.0)  # Reduced from 0.001
        vth_effective = self.vth_mismatch + temp_effect
        
        # Voltage effect on current (reduced sensitivity)
        voltage_nominal = 1.2
        voltage_ratio = self.environmental_stressors['voltage'] / voltage_nominal
        beta_effective = self.beta_mismatch * voltage_ratio
        
        # Calculate strength ratio between transistor pairs
        strength_ratio = (beta_effective[:,:,0] * (1 - vth_effective[:,:,0])) / \
                        (beta_effective[:,:,1] * (1 - vth_effective[:,:,1]))
        
        # Add noise effect (reduced sensitivity)
        noise = np.random.normal(0, self.environmental_stressors['em_noise'] * 0.025, size=(self.rows, self.columns))  # Reduced scale
        strength_ratio += noise
        
        return (strength_ratio > 1).astype(int)

    def generate_crps(self, num_crps: int) -> np.ndarray:
        """Generate responses (startup states) under current conditions.
        
        Args:
            num_crps: Number of startup patterns to generate
        """
        responses = np.array([self.generate_startup_state().flatten() for _ in range(num_crps)])
        return responses

class RingOscillatorPUF(PUF):
    def __init__(
        self,
        num_oscillators: int,
        stages_per_oscillator: int = 13,
        seed: Optional[int] = None,
        environmental_stressors: Optional[Dict[str, float]] = None
    ):
        """Initialize Ring Oscillator PUF.
        
        Args:
            num_oscillators: Number of ring oscillators
            stages_per_oscillator: Number of inverter stages per oscillator
            seed: Random seed for reproducibility
            environmental_stressors: Environmental conditions
            
        References:
            - Suh and Devadas "Physical Unclonable Functions for Device Authentication" (DAC 2007)
            - Maiti and Schaumont "Improved Ring Oscillator PUF" (HOST 2011)
        """
        super().__init__(seed, environmental_stressors)
        self.num_oscillators = num_oscillators
        self.stages_per_oscillator = stages_per_oscillator
        
        # Model oscillator characteristics
        self.base_freq = 100e6  # 100 MHz base frequency
        self.process_variations = self.generate_process_variations()

    def generate_process_variations(self) -> np.ndarray:
        """Generate process variation effects for each oscillator.
        
        Models:
        - Global process variations
        - Local variations per stage
        - Spatial correlation
        """
        # Global variations (shared across nearby oscillators)
        spatial_correlation = 0.7
        position_x = np.random.uniform(0, 1, self.num_oscillators)
        position_y = np.random.uniform(0, 1, self.num_oscillators)
        
        # Calculate spatial correlation matrix
        dist_matrix = np.zeros((self.num_oscillators, self.num_oscillators))
        for i in range(self.num_oscillators):
            for j in range(self.num_oscillators):
                dist = np.sqrt((position_x[i] - position_x[j])**2 + (position_y[i] - position_y[j])**2)
                dist_matrix[i,j] = np.exp(-dist / spatial_correlation)
        
        # Generate correlated variations (reduced variance)
        global_variations = np.random.multivariate_normal(
            mean=np.zeros(self.num_oscillators),
            cov=dist_matrix * 0.005  # Reduced from 0.01
        )
        
        # Local variations per oscillator (reduced variance)
        local_variations = np.random.normal(0, 0.0025, self.num_oscillators)  # Reduced from 0.005
        
        return global_variations + local_variations

    def get_frequency(self, oscillator_idx: int) -> float:
        """Calculate frequency for a specific oscillator.
        
        Models:
        - Process variations
        - Temperature effects
        - Voltage effects
        - Noise
        """
        # Base frequency with process variations
        freq = self.base_freq * (1 + self.process_variations[oscillator_idx])
        
        # Temperature effect (reduced sensitivity)
        temp_effect = 1.0 - 0.0005 * (self.environmental_stressors['temperature'] - 25.0)  # Reduced from 0.001
        
        # Voltage effect (reduced sensitivity)
        voltage_nominal = 1.2
        voltage_ratio = self.environmental_stressors['voltage'] / voltage_nominal
        voltage_effect = voltage_ratio ** 1.5  # Reduced from quadratic
        
        # Apply environmental effects
        freq *= temp_effect * voltage_effect
        
        # Add noise (reduced sensitivity)
        noise = np.random.normal(0, self.environmental_stressors['em_noise'] * 0.025 * freq)  # Reduced scale
        freq += noise
        
        return freq

    def evaluate(self, challenge: Tuple[int, int]) -> int:
        """Compare frequencies of two oscillators.
        
        Args:
            challenge: Tuple of (oscillator_1_idx, oscillator_2_idx)
        """
        freq1 = self.get_frequency(challenge[0])
        freq2 = self.get_frequency(challenge[1])
        return 1 if freq1 > freq2 else 0

    def generate_crps(
        self,
        num_crps: int,
        challenges: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Generate challenge-response pairs.
        
        Args:
            num_crps: Number of CRPs to generate
            challenges: Optional predefined challenges
        """
        if challenges is None:
            challenges = [
                tuple(np.random.choice(self.num_oscillators, 2, replace=False))
                for _ in range(num_crps)
            ]
        
        responses = np.array([self.evaluate(ch) for ch in challenges])
        return challenges, responses 