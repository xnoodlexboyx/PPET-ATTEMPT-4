import numpy as np
from typing import Dict, List, Tuple, Union
from .puf_emulator import PUF
from ..utilities.data_generators import DataGenerator

class PUFAnalyzer:
    def __init__(self, puf: PUF):
        """Initialize PUF analyzer.
        
        Args:
            puf: PUF instance to analyze
        """
        self.puf = puf
        self.data_generator = DataGenerator()
        self.analysis_results = {}

    def analyze_uniqueness(
        self,
        num_instances: int,
        num_challenges: int,
        puf_params: Dict
    ) -> Dict:
        """Analyze inter-device uniqueness of PUF responses.
        
        Args:
            num_instances: Number of PUF instances to compare
            num_challenges: Number of challenges per instance
            puf_params: Parameters for PUF initialization
        
        Returns:
            Dictionary containing uniqueness metrics
        """
        # Generate population of PUF instances
        puf_type = self.puf.__class__.__name__.lower().replace('puf', '')
        population = self.data_generator.generate_puf_population(
            puf_type,
            num_instances,
            puf_params
        )
        
        # Generate responses for each instance
        responses = []
        for puf_instance in population:
            if hasattr(puf_instance, 'generate_crps'):
                _, instance_responses = puf_instance.generate_crps(num_challenges)
            else:
                instance_responses = puf_instance.generate_crps(num_challenges)
            responses.append(instance_responses)
        
        responses = np.array(responses)
        
        # Calculate Hamming distances between all pairs
        n = len(responses)
        hamming_distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.mean(responses[i] != responses[j])
                hamming_distances.append(distance)
        
        # Calculate metrics
        avg_distance = np.mean(hamming_distances)
        std_distance = np.std(hamming_distances)
        
        # Ideal uniqueness is 0.5 (50% difference between instances)
        uniqueness_quality = 1 - 2 * abs(0.5 - avg_distance)
        
        metrics = {
            'average_hamming_distance': avg_distance,
            'std_hamming_distance': std_distance,
            'uniqueness_quality': uniqueness_quality,
            'num_instances': num_instances,
            'num_challenges': num_challenges
        }
        
        self.analysis_results['uniqueness'] = metrics
        return metrics

    def analyze_reliability(
        self,
        num_crps: int,
        env_conditions: Dict[str, Dict]
    ) -> Dict:
        """Analyze reliability under environmental variations.
        
        Args:
            num_crps: Number of CRPs to test
            env_conditions: Environmental conditions to test
                Example: {
                    'temperature': {'min': -40, 'max': 85, 'points': 100},
                    'voltage': {'nominal': 1.2, 'variation': 0.1, 'points': 50}
                }
        
        Returns:
            Dictionary containing reliability metrics
        """
        # Generate reliability dataset
        nominal_responses, condition_responses, env_data = \
            self.data_generator.generate_reliability_dataset(
                self.puf,
                num_crps,
                env_conditions
            )
        
        # Calculate reliability metrics for each condition
        reliability_metrics = {}
        
        for condition, responses in condition_responses.items():
            # Calculate bit error rate for each environmental value
            bit_error_rates = []
            for response in responses:
                bit_errors = np.mean(response != nominal_responses)
                bit_error_rates.append(bit_errors)
            
            # Calculate metrics
            avg_ber = np.mean(bit_error_rates)
            max_ber = np.max(bit_error_rates)
            std_ber = np.std(bit_error_rates)
            
            reliability_metrics[condition] = {
                'average_bit_error_rate': avg_ber,
                'max_bit_error_rate': max_ber,
                'std_bit_error_rate': std_ber,
                'environmental_values': env_data[condition]
            }
        
        # Overall reliability score (lower is better)
        overall_reliability = np.mean([
            metrics['average_bit_error_rate']
            for metrics in reliability_metrics.values()
        ])
        
        metrics = {
            'condition_metrics': reliability_metrics,
            'overall_reliability': overall_reliability,
            'num_crps': num_crps
        }
        
        self.analysis_results['reliability'] = metrics
        return metrics

    def analyze_bit_aliasing(
        self,
        num_instances: int,
        num_challenges: int,
        puf_params: Dict
    ) -> Dict:
        """Analyze bit aliasing across PUF population.
        
        Args:
            num_instances: Number of PUF instances to analyze
            num_challenges: Number of challenges per instance
            puf_params: Parameters for PUF initialization
        
        Returns:
            Dictionary containing bit aliasing metrics
        """
        # Generate population of PUF instances
        puf_type = self.puf.__class__.__name__.lower().replace('puf', '')
        population = self.data_generator.generate_puf_population(
            puf_type,
            num_instances,
            puf_params
        )
        
        # Generate responses for each instance
        responses = []
        for puf_instance in population:
            if hasattr(puf_instance, 'generate_crps'):
                _, instance_responses = puf_instance.generate_crps(num_challenges)
            else:
                instance_responses = puf_instance.generate_crps(num_challenges)
            responses.append(instance_responses)
        
        responses = np.array(responses)
        
        # Calculate probability of '1' for each response bit
        prob_ones = np.mean(responses, axis=0)
        
        # Calculate bit aliasing metrics
        avg_prob = np.mean(prob_ones)
        std_prob = np.std(prob_ones)
        max_bias = np.max(np.abs(prob_ones - 0.5))
        
        # Calculate uniformity score (ideal is 0.5)
        uniformity = 1 - 2 * abs(0.5 - avg_prob)
        
        metrics = {
            'average_probability': avg_prob,
            'std_probability': std_prob,
            'max_bias': max_bias,
            'uniformity_score': uniformity,
            'num_instances': num_instances,
            'num_challenges': num_challenges
        }
        
        self.analysis_results['bit_aliasing'] = metrics
        return metrics

    def analyze_all(
        self,
        num_instances: int,
        num_challenges: int,
        puf_params: Dict,
        env_conditions: Dict[str, Dict]
    ) -> Dict:
        """Perform comprehensive PUF analysis.
        
        Args:
            num_instances: Number of PUF instances to analyze
            num_challenges: Number of challenges per instance
            puf_params: Parameters for PUF initialization
            env_conditions: Environmental conditions to test
        
        Returns:
            Dictionary containing all analysis metrics
        """
        uniqueness = self.analyze_uniqueness(
            num_instances,
            num_challenges,
            puf_params
        )
        
        reliability = self.analyze_reliability(
            num_challenges,
            env_conditions
        )
        
        bit_aliasing = self.analyze_bit_aliasing(
            num_instances,
            num_challenges,
            puf_params
        )
        
        # Calculate overall quality score (simplified)
        quality_score = (
            uniqueness['uniqueness_quality'] * 0.4 +
            (1 - reliability['overall_reliability']) * 0.4 +
            bit_aliasing['uniformity_score'] * 0.2
        )
        
        comprehensive_metrics = {
            'uniqueness': uniqueness,
            'reliability': reliability,
            'bit_aliasing': bit_aliasing,
            'overall_quality_score': quality_score
        }
        
        self.analysis_results['comprehensive'] = comprehensive_metrics
        return comprehensive_metrics 