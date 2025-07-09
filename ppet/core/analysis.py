import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

class PUFAnalyzer:
    """Analyzer for PUF characteristics and metrics."""
    
    def __init__(self):
        """Initialize PUF analyzer."""
        self.metrics = {}
        self.challenge_length = None
        self.num_instances = None
    
    def analyze_uniqueness(
        self,
        responses: np.ndarray,
        challenges: Optional[np.ndarray] = None
    ) -> float:
        """Calculate inter-chip hamming distance (uniqueness).
        
        Args:
            responses: Response matrix (num_instances x num_challenges)
            challenges: Optional challenge matrix for correlation analysis
        
        Returns:
            Average inter-chip hamming distance percentage
        """
        num_instances, num_challenges = responses.shape
        total_comparisons = 0
        total_hd = 0
        
        # Compare each pair of instances
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                hd = np.sum(responses[i] != responses[j])
                total_hd += hd
                total_comparisons += num_challenges
        
        uniqueness = (total_hd / total_comparisons) * 100
        self.metrics['uniqueness'] = uniqueness
        
        if challenges is not None:
            # Analyze response correlation with challenges
            challenge_correlation = np.corrcoef(challenges.T, responses.T)
            self.metrics['challenge_correlation'] = challenge_correlation
        
        return uniqueness
    
    def analyze_reliability(
        self,
        responses: np.ndarray,
        noise_responses: np.ndarray
    ) -> float:
        """Calculate bit error rate under noise (reliability).
        
        Args:
            responses: Original response matrix (num_instances x num_challenges)
            noise_responses: Noisy response matrix
        
        Returns:
            Average bit error rate percentage
        """
        if responses.shape != noise_responses.shape:
            raise ValueError("Response matrices must have same shape")
        
        # Calculate bit errors
        bit_errors = np.sum(responses != noise_responses, axis=1)
        total_bits = responses.shape[1]
        
        # Calculate BER for each instance
        ber = (bit_errors / total_bits) * 100
        avg_ber = np.mean(ber)
        std_ber = np.std(ber)
        
        self.metrics['reliability'] = 100 - avg_ber
        self.metrics['reliability_std'] = std_ber
        
        return 100 - avg_ber  # Convert to reliability percentage
    
    def analyze_bit_aliasing(self, responses: np.ndarray) -> float:
        """Calculate bit aliasing across PUF instances.
        
        Args:
            responses: Response matrix (num_instances x num_challenges)
        
        Returns:
            Average bit aliasing percentage
        """
        num_instances = responses.shape[0]
        
        # Calculate probability of 1s for each challenge
        prob_ones = np.mean(responses, axis=0)
        
        # Calculate bit aliasing (deviation from ideal 50%)
        bit_aliasing = float(np.mean(np.abs(prob_ones - 0.5)) * 100)
        
        self.metrics['bit_aliasing'] = bit_aliasing
        self.metrics['bit_bias'] = float(np.mean(prob_ones))
        
        return bit_aliasing
    
    def analyze_entropy(self, responses: np.ndarray) -> float:
        """Calculate response entropy.
        
        Args:
            responses: Response matrix (num_instances x num_challenges)
        
        Returns:
            Average response entropy (bits)
        """
        # Calculate response probabilities
        response_counts = np.bincount(responses.flatten())
        response_probs = response_counts / len(responses.flatten())
        
        # Calculate Shannon entropy
        response_entropy = entropy(response_probs, base=2)
        
        self.metrics['entropy'] = response_entropy
        return response_entropy
    
    def analyze_uniformity(self, responses: np.ndarray) -> float:
        """Calculate response uniformity.
        
        Args:
            responses: Response matrix (num_instances x num_challenges)
        
        Returns:
            Uniformity percentage
        """
        # Calculate percentage of 1s
        uniformity = np.mean(responses) * 100
        
        self.metrics['uniformity'] = uniformity
        return uniformity
    
    def plot_metrics(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot PUF metrics visualization.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Create bar plot of main metrics
        metrics_to_plot = [
            'uniqueness',
            'reliability',
            'bit_aliasing',
            'uniformity'
        ]
        
        values = [self.metrics.get(m, 0) for m in metrics_to_plot]
        
        plt.bar(metrics_to_plot, values)
        plt.axhline(y=50, color='r', linestyle='--', label='Ideal')
        
        plt.title('PUF Quality Metrics')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report.
        
        Returns:
            Dictionary containing all metrics and analysis results
        """
        report = {
            'summary': {
                metric: f"{value:.2f}" if isinstance(value, float) else value
                for metric, value in self.metrics.items()
            },
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if self.metrics.get('uniqueness', 0) < 45:
            report['recommendations'].append(
                "Uniqueness below target (45-55%). Consider increasing "
                "manufacturing variation parameters."
            )
        
        if self.metrics.get('reliability', 100) < 95:
            report['recommendations'].append(
                "Reliability below 95%. Consider reducing noise sensitivity "
                "or environmental variation."
            )
        
        if self.metrics.get('bit_aliasing', 0) > 10:
            report['recommendations'].append(
                "High bit aliasing (>10%). Check for systematic bias in "
                "the PUF design."
            )
        
        if abs(self.metrics.get('uniformity', 50) - 50) > 5:
            report['recommendations'].append(
                "Response uniformity deviates >5% from ideal 50%. "
                "Check for response bias."
            )
        
        return report 