import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from ..core.puf_emulator import PUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF

class DataGenerator:
    def __init__(self, seed: Optional[int] = None):
        """Initialize the data generator with an optional seed for reproducibility."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_crps(self, puf: PUF, num_crps: int) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Generate challenge-response pairs for a given PUF.
        
        Args:
            puf: Instance of a PUF class
            num_crps: Number of challenge-response pairs to generate
            
        Returns:
            For Arbiter and RO PUFs: (challenges, responses)
            For SRAM PUF: responses only (as challenges don't apply)
        """
        return puf.generate_crps(num_crps)

    def generate_environmental_data(self, conditions: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Generate environmental stressor data based on specified conditions.
        
        Args:
            conditions: Dictionary specifying environmental conditions and their parameters
                Example: {
                    'temperature': {'min': -40, 'max': 85, 'points': 100},
                    'voltage': {'nominal': 1.2, 'variation': 0.1, 'points': 50},
                    'em_noise': {'mean': 0, 'std': 1, 'points': 75}
                }
        
        Returns:
            Dictionary containing generated environmental data
        """
        env_data = {}
        
        for condition, params in conditions.items():
            if condition == 'temperature':
                # Linear temperature range
                env_data[condition] = np.linspace(
                    params['min'],
                    params['max'],
                    params['points']
                )
            
            elif condition == 'voltage':
                # Normal distribution around nominal voltage
                env_data[condition] = np.random.normal(
                    params['nominal'],
                    params['variation'],
                    params['points']
                )
            
            elif condition == 'em_noise':
                # Gaussian noise for electromagnetic interference
                env_data[condition] = np.random.normal(
                    params['mean'],
                    params['std'],
                    params['points']
                )
            
            else:
                raise ValueError(f"Unsupported environmental condition: {condition}")
        
        return env_data

    def generate_puf_population(
        self,
        puf_type: str,
        num_instances: int,
        params: Dict
    ) -> List[PUF]:
        """Generate a population of PUF instances for statistical analysis.
        
        Args:
            puf_type: Type of PUF ('arbiter', 'sram', or 'ro')
            num_instances: Number of PUF instances to generate
            params: Parameters for PUF initialization
                Example for Arbiter PUF: {'n_stages': 64}
                Example for SRAM PUF: {'rows': 128, 'columns': 128}
                Example for RO PUF: {'num_oscillators': 128}
        
        Returns:
            List of PUF instances
        """
        puf_classes = {
            'arbiter': ArbiterPUF,
            'sram': SRAMPUF,
            'ro': RingOscillatorPUF
        }
        
        if puf_type not in puf_classes:
            raise ValueError(f"Unsupported PUF type: {puf_type}")
        
        puf_class = puf_classes[puf_type]
        population = []
        
        for i in range(num_instances):
            # Use different seed for each instance
            instance_seed = None if self.seed is None else self.seed + i
            puf_instance = puf_class(seed=instance_seed, **params)
            population.append(puf_instance)
        
        return population

    def generate_reliability_dataset(
        self,
        puf: PUF,
        num_crps: int,
        env_conditions: Dict[str, Dict]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate dataset for reliability analysis under various conditions.
        
        Args:
            puf: PUF instance
            num_crps: Number of CRPs to generate
            env_conditions: Environmental conditions to test
        
        Returns:
            (nominal_responses, condition_responses, environmental_data)
        """
        # Generate nominal responses
        if isinstance(puf, SRAMPUF):
            nominal_responses = puf.generate_crps(num_crps)
        else:
            challenges, nominal_responses = puf.generate_crps(num_crps)
        
        # Generate environmental data
        env_data = self.generate_environmental_data(env_conditions)
        
        # Store original environmental stressors to restore later
        original_stressors = puf.environmental_stressors.copy()
        
        # Generate responses under each condition
        condition_responses = {}
        for condition, values in env_data.items():
            responses_under_condition = []
            for value in values:
                # Update only the specific environmental stressor
                puf.environmental_stressors[condition] = value
                if isinstance(puf, SRAMPUF):
                    responses = puf.generate_crps(num_crps)
                else:
                    _, responses = puf.generate_crps(num_crps)
                responses_under_condition.append(responses)
            condition_responses[condition] = np.array(responses_under_condition)
            
        # Restore original environmental stressors
        puf.environmental_stressors = original_stressors
        
        return nominal_responses, condition_responses, env_data 