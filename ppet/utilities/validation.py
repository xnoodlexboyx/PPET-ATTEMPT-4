"""
Validation framework for PPET models and simulations.

This module provides comprehensive validation and verification tools
for ensuring the accuracy and reliability of PUF simulations.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from ..core.puf_emulator import PUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF

@dataclass
class ValidationResult:
    """Container for validation results."""
    test_name: str
    passed: bool
    score: float
    confidence_interval: Tuple[float, float]
    p_value: float
    details: Dict[str, Any]

class PUFValidator:
    """Comprehensive PUF validation framework."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize validator.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results = []
    
    def validate_uniqueness(
        self,
        pufs: List[PUF],
        num_challenges: int = 1000,
        expected_mean: float = 0.5,
        tolerance: float = 0.05
    ) -> ValidationResult:
        """Validate PUF uniqueness properties.
        
        Args:
            pufs: List of PUF instances
            num_challenges: Number of challenges to test
            expected_mean: Expected mean Hamming distance
            tolerance: Acceptable tolerance from expected mean
            
        Returns:
            ValidationResult object
        """
        # Generate challenges
        challenges = np.random.randint(0, 2, size=(num_challenges, pufs[0].n_stages))
        
        # Collect responses
        responses = []
        for puf in pufs:
            puf_responses = []
            for challenge in challenges:
                response = puf.evaluate(challenge)
                puf_responses.append(response)
            responses.append(puf_responses)
        
        responses = np.array(responses)
        
        # Calculate pairwise Hamming distances
        n_pufs = len(pufs)
        hamming_distances = []
        
        for i in range(n_pufs):
            for j in range(i + 1, n_pufs):
                distance = np.mean(responses[i] != responses[j])
                hamming_distances.append(distance)
        
        hamming_distances = np.array(hamming_distances)
        
        # Statistical analysis
        mean_distance = np.mean(hamming_distances)
        std_distance = np.std(hamming_distances)
        
        # Confidence interval for mean
        n_samples = len(hamming_distances)
        sem = std_distance / np.sqrt(n_samples)
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n_samples - 1)
        ci_lower = mean_distance - t_critical * sem
        ci_upper = mean_distance + t_critical * sem
        
        # Test if mean is within tolerance of expected value
        t_stat = (mean_distance - expected_mean) / sem
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 1))
        
        # Check if within tolerance
        passed = abs(mean_distance - expected_mean) <= tolerance
        
        # Normality test
        _, normality_p = stats.shapiro(hamming_distances)
        
        result = ValidationResult(
            test_name="Uniqueness Validation",
            passed=passed,
            score=1 - abs(mean_distance - expected_mean) / tolerance,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            details={
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'expected_mean': expected_mean,
                'tolerance': tolerance,
                'n_samples': n_samples,
                'normality_p_value': normality_p,
                'raw_distances': hamming_distances
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_reliability(
        self,
        puf: PUF,
        num_challenges: int = 500,
        num_trials: int = 100,
        min_reliability: float = 0.95
    ) -> ValidationResult:
        """Validate PUF reliability under repeated evaluation.
        
        Args:
            puf: PUF instance to test
            num_challenges: Number of challenges to test
            num_trials: Number of trials per challenge
            min_reliability: Minimum acceptable reliability
            
        Returns:
            ValidationResult object
        """
        # Generate challenges
        challenges = np.random.randint(0, 2, size=(num_challenges, puf.n_stages))
        
        # Test reliability for each challenge
        reliability_scores = []
        
        for challenge in challenges:
            # Get reference response
            ref_response = puf.evaluate(challenge)
            
            # Repeated evaluations
            responses = []
            for _ in range(num_trials):
                response = puf.evaluate(challenge)
                responses.append(response)
            
            # Calculate reliability for this challenge
            reliability = np.mean(np.array(responses) == ref_response)
            reliability_scores.append(reliability)
        
        reliability_scores = np.array(reliability_scores)
        
        # Statistical analysis
        mean_reliability = np.mean(reliability_scores)
        std_reliability = np.std(reliability_scores)
        
        # Confidence interval
        n_samples = len(reliability_scores)
        sem = std_reliability / np.sqrt(n_samples)
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n_samples - 1)
        ci_lower = mean_reliability - t_critical * sem
        ci_upper = mean_reliability + t_critical * sem
        
        # Test if reliability meets minimum requirement
        t_stat = (mean_reliability - min_reliability) / sem
        p_value = 1 - stats.t.cdf(t_stat, n_samples - 1)
        
        passed = mean_reliability >= min_reliability
        
        result = ValidationResult(
            test_name="Reliability Validation",
            passed=passed,
            score=mean_reliability,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            details={
                'mean_reliability': mean_reliability,
                'std_reliability': std_reliability,
                'min_reliability': min_reliability,
                'n_samples': n_samples,
                'raw_scores': reliability_scores
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_bit_aliasing(
        self,
        pufs: List[PUF],
        num_challenges: int = 1000,
        max_bias: float = 0.1
    ) -> ValidationResult:
        """Validate bit aliasing properties.
        
        Args:
            pufs: List of PUF instances
            num_challenges: Number of challenges to test
            max_bias: Maximum acceptable bias from 0.5
            
        Returns:
            ValidationResult object
        """
        # Generate challenges
        challenges = np.random.randint(0, 2, size=(num_challenges, pufs[0].n_stages))
        
        # Collect responses
        responses = []
        for puf in pufs:
            puf_responses = []
            for challenge in challenges:
                response = puf.evaluate(challenge)
                puf_responses.append(response)
            responses.append(puf_responses)
        
        responses = np.array(responses)
        
        # Calculate bit probabilities
        bit_probs = np.mean(responses, axis=0)
        
        # Calculate bias from ideal 0.5
        bias_values = np.abs(bit_probs - 0.5)
        max_observed_bias = np.max(bias_values)
        mean_bias = np.mean(bias_values)
        
        # Statistical test for uniformity
        # Chi-square test for each bit position
        chi2_stats = []
        for i in range(len(bit_probs)):
            ones = int(bit_probs[i] * len(pufs))
            zeros = len(pufs) - ones
            chi2_stat = ((ones - len(pufs)/2)**2 + (zeros - len(pufs)/2)**2) / (len(pufs)/2)
            chi2_stats.append(chi2_stat)
        
        # Overall chi-square test
        overall_chi2 = np.sum(chi2_stats)
        df = len(bit_probs)
        chi2_p_value = 1 - stats.chi2.cdf(overall_chi2, df)
        
        # Check if bias is within acceptable range
        passed = max_observed_bias <= max_bias
        
        result = ValidationResult(
            test_name="Bit-Aliasing Validation",
            passed=passed,
            score=1 - max_observed_bias / max_bias if max_bias > 0 else 0,
            confidence_interval=(mean_bias - np.std(bias_values), mean_bias + np.std(bias_values)),
            p_value=chi2_p_value,
            details={
                'max_bias': max_observed_bias,
                'mean_bias': mean_bias,
                'acceptable_bias': max_bias,
                'bit_probabilities': bit_probs,
                'chi2_statistic': overall_chi2,
                'degrees_of_freedom': df
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_environmental_response(
        self,
        puf: PUF,
        temperature_range: Tuple[float, float] = (-40, 85),
        voltage_range: Tuple[float, float] = (1.0, 1.4),
        num_challenges: int = 100
    ) -> ValidationResult:
        """Validate PUF response to environmental conditions.
        
        Args:
            puf: PUF instance to test
            temperature_range: Temperature range to test (°C)
            voltage_range: Voltage range to test (V)
            num_challenges: Number of challenges to test
            
        Returns:
            ValidationResult object
        """
        # Generate challenges
        challenges = np.random.randint(0, 2, size=(num_challenges, puf.n_stages))
        
        # Test conditions
        test_conditions = [
            {'temperature': temperature_range[0], 'voltage': voltage_range[0]},
            {'temperature': temperature_range[1], 'voltage': voltage_range[1]},
            {'temperature': 25.0, 'voltage': 1.2},  # Nominal
        ]
        
        # Collect responses under different conditions
        condition_responses = []
        for condition in test_conditions:
            # Set environmental conditions
            original_stressors = puf.environmental_stressors.copy()
            puf.environmental_stressors.update(condition)
            
            # Collect responses
            responses = []
            for challenge in challenges:
                response = puf.evaluate(challenge)
                responses.append(response)
            
            condition_responses.append(responses)
            
            # Restore original conditions
            puf.environmental_stressors = original_stressors
        
        # Calculate sensitivity to environmental conditions
        nominal_responses = np.array(condition_responses[2])
        
        sensitivities = []
        for i in range(2):  # Compare extreme conditions to nominal
            test_responses = np.array(condition_responses[i])
            sensitivity = np.mean(test_responses != nominal_responses)
            sensitivities.append(sensitivity)
        
        mean_sensitivity = np.mean(sensitivities)
        max_sensitivity = np.max(sensitivities)
        
        # Consider passed if sensitivity is reasonable but not too high
        passed = 0.05 <= mean_sensitivity <= 0.3  # 5-30% change expected
        
        result = ValidationResult(
            test_name="Environmental Response Validation",
            passed=passed,
            score=1 - abs(mean_sensitivity - 0.175) / 0.175,  # Optimal around 17.5%
            confidence_interval=(min(sensitivities), max(sensitivities)),
            p_value=0.0,  # Not applicable for this test
            details={
                'mean_sensitivity': mean_sensitivity,
                'max_sensitivity': max_sensitivity,
                'individual_sensitivities': sensitivities,
                'test_conditions': test_conditions,
                'expected_range': (0.05, 0.3)
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_model_parameters(
        self,
        puf: PUF,
        expected_params: Dict[str, Tuple[float, float]]
    ) -> ValidationResult:
        """Validate model parameters against expected ranges.
        
        Args:
            puf: PUF instance to validate
            expected_params: Dictionary of parameter name -> (min, max) ranges
            
        Returns:
            ValidationResult object
        """
        violations = []
        param_scores = []
        
        for param_name, (min_val, max_val) in expected_params.items():
            if hasattr(puf, param_name):
                param_value = getattr(puf, param_name)
                
                if isinstance(param_value, np.ndarray):
                    param_value = np.mean(param_value)
                
                # Check if parameter is within expected range
                if param_value < min_val or param_value > max_val:
                    violations.append(f"{param_name}: {param_value} not in [{min_val}, {max_val}]")
                    param_scores.append(0)
                else:
                    # Score based on how close to center of range
                    center = (min_val + max_val) / 2
                    width = max_val - min_val
                    score = 1 - abs(param_value - center) / (width / 2)
                    param_scores.append(score)
        
        overall_score = np.mean(param_scores) if param_scores else 0
        passed = len(violations) == 0
        
        result = ValidationResult(
            test_name="Model Parameter Validation",
            passed=passed,
            score=overall_score,
            confidence_interval=(0.0, 1.0),  # Not applicable
            p_value=0.0,  # Not applicable
            details={
                'violations': violations,
                'parameter_scores': dict(zip(expected_params.keys(), param_scores)),
                'expected_parameters': expected_params
            }
        )
        
        self.results.append(result)
        return result
    
    def run_comprehensive_validation(
        self,
        pufs: List[PUF],
        num_challenges: int = 1000
    ) -> Dict[str, ValidationResult]:
        """Run comprehensive validation suite.
        
        Args:
            pufs: List of PUF instances to validate
            num_challenges: Number of challenges for tests
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Uniqueness validation
        results['uniqueness'] = self.validate_uniqueness(pufs, num_challenges)
        
        # Reliability validation
        results['reliability'] = self.validate_reliability(pufs[0], num_challenges // 2)
        
        # Bit-aliasing validation
        results['bit_aliasing'] = self.validate_bit_aliasing(pufs, num_challenges)
        
        # Environmental response validation
        results['environmental'] = self.validate_environmental_response(pufs[0])
        
        # Model parameter validation
        if isinstance(pufs[0], ArbiterPUF):
            expected_params = {
                'global_variation': (-1.0, 1.0),
                'arbiter_bias': (-0.5, 0.5),
            }
            results['parameters'] = self.validate_model_parameters(pufs[0], expected_params)
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report.
        
        Returns:
            Formatted validation report
        """
        if not self.results:
            return "No validation results available."
        
        report = []
        report.append("PPET Validation Report")
        report.append("=" * 50)
        report.append(f"Total tests run: {len(self.results)}")
        
        passed_tests = sum(1 for r in self.results if r.passed)
        report.append(f"Passed tests: {passed_tests}")
        report.append(f"Failed tests: {len(self.results) - passed_tests}")
        report.append(f"Success rate: {passed_tests / len(self.results) * 100:.1f}%")
        report.append("")
        
        # Detailed results
        for result in self.results:
            report.append(f"Test: {result.test_name}")
            report.append(f"  Status: {'PASS' if result.passed else 'FAIL'}")
            report.append(f"  Score: {result.score:.3f}")
            report.append(f"  Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
            report.append(f"  P-value: {result.p_value:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    def clear_results(self):
        """Clear all validation results."""
        self.results = []

class CrossValidator:
    """Cross-validation framework for PUF models."""
    
    def __init__(self, k_folds: int = 5):
        """Initialize cross-validator.
        
        Args:
            k_folds: Number of folds for cross-validation
        """
        self.k_folds = k_folds
    
    def cross_validate_attack(
        self,
        attack_class,
        attack_params: Dict,
        pufs: List[PUF],
        num_challenges: int = 1000
    ) -> Dict[str, float]:
        """Cross-validate attack performance.
        
        Args:
            attack_class: Attack class to test
            attack_params: Parameters for attack initialization
            pufs: List of PUF instances
            num_challenges: Number of challenges per fold
            
        Returns:
            Cross-validation results
        """
        fold_scores = []
        
        for fold in range(self.k_folds):
            # Create attack instance
            attack = attack_class(**attack_params)
            
            # Generate training and test data
            train_size = int(num_challenges * 0.8)
            
            # Generate challenges and responses
            challenges = np.random.randint(0, 2, size=(num_challenges, pufs[0].n_stages))
            responses = []
            
            for challenge in challenges:
                response = pufs[fold % len(pufs)].evaluate(challenge)
                responses.append(response)
            
            responses = np.array(responses)
            
            # Split data
            train_challenges = challenges[:train_size]
            train_responses = responses[:train_size]
            test_challenges = challenges[train_size:]
            test_responses = responses[train_size:]
            
            # Train and evaluate
            try:
                attack.train(train_challenges, train_responses)
                score = attack.evaluate(test_challenges, test_responses)
                fold_scores.append(score)
            except Exception as e:
                warnings.warn(f"Fold {fold} failed: {e}")
                fold_scores.append(0.0)
        
        return {
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'fold_scores': fold_scores,
            'confidence_interval': (
                np.mean(fold_scores) - 1.96 * np.std(fold_scores) / np.sqrt(self.k_folds),
                np.mean(fold_scores) + 1.96 * np.std(fold_scores) / np.sqrt(self.k_folds)
            )
        }