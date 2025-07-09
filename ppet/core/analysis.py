"""PUF analysis and visualization module with military-grade capabilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from .puf_emulator import PUF
from .military_stressors import MilitaryEnvironment, MilitaryStressors

class PUFAnalyzer:
    def __init__(self, puf: PUF):
        """Initialize PUF analyzer.
        
        Args:
            puf: PUF instance to analyze
        """
        self.puf = puf
        
    def analyze_reliability_under_stress(
        self,
        challenge: np.ndarray,
        num_trials: int = 100,
        time_points: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Analyze PUF reliability under military environmental stress.
        
        Args:
            challenge: Challenge bits to evaluate
            num_trials: Number of trials per time point
            time_points: List of mission times to evaluate (hours)
            
        Returns:
            Dictionary containing reliability metrics
        """
        if time_points is None:
            time_points = np.linspace(0, 1000, 20)  # Default 1000-hour mission
            
        responses = []
        temperatures = []
        emi_levels = []
        aging_factors = []
        
        base_response = self.puf.evaluate(challenge)
        
        for time in time_points:
            self.puf.update_mission_time(time)
            
            trial_responses = []
            for _ in range(num_trials):
                response = self.puf.evaluate(challenge)
                trial_responses.append(response)
                
            responses.append(trial_responses)
            temperatures.append(self.puf.environmental_stressors['temperature'])
            emi_levels.append(self.puf.environmental_stressors['em_noise'])
            aging_factors.append(self.puf.environmental_stressors.get('aging_factor', 1.0))
            
        return {
            'time_points': time_points,
            'responses': np.array(responses),
            'temperatures': np.array(temperatures),
            'emi_levels': np.array(emi_levels),
            'aging_factors': np.array(aging_factors),
            'base_response': base_response
        }
        
    def plot_reliability_analysis(
        self,
        analysis_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """Create comprehensive reliability visualization.
        
        Args:
            analysis_data: Data from analyze_reliability_under_stress
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)
        
        # Reliability over time
        ax1 = fig.add_subplot(gs[0, :])
        reliability = np.mean(analysis_data['responses'] == analysis_data['base_response'], axis=1)
        ax1.plot(analysis_data['time_points'], reliability * 100, 'b-', label='Reliability')
        ax1.set_xlabel('Mission Time (hours)')
        ax1.set_ylabel('Reliability (%)')
        ax1.set_title('PUF Reliability Over Mission Time')
        ax1.grid(True)
        
        # Temperature profile
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(analysis_data['time_points'], analysis_data['temperatures'], 'r-')
        ax2.set_xlabel('Mission Time (hours)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Profile')
        ax2.grid(True)
        
        # EMI profile
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(analysis_data['time_points'], analysis_data['emi_levels'], 'g-')
        ax3.set_xlabel('Mission Time (hours)')
        ax3.set_ylabel('EMI Level (normalized)')
        ax3.set_title('EMI Profile')
        ax3.grid(True)
        
        # Response distribution
        ax4 = fig.add_subplot(gs[2, 0])
        flipped_bits = np.sum(analysis_data['responses'] != analysis_data['base_response'], axis=0)
        sns.histplot(data=flipped_bits, ax=ax4, bins=20)
        ax4.set_xlabel('Number of Bit Flips')
        ax4.set_ylabel('Count')
        ax4.set_title('Response Stability Distribution')
        
        # Aging effects
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(analysis_data['time_points'], analysis_data['aging_factors'], 'm-')
        ax5.set_xlabel('Mission Time (hours)')
        ax5.set_ylabel('Aging Factor')
        ax5.set_title('Aging Degradation')
        ax5.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def analyze_environmental_sensitivity(
        self,
        challenge: np.ndarray,
        environment: MilitaryEnvironment,
        num_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Analyze PUF sensitivity to environmental conditions.
        
        Args:
            challenge: Challenge bits to evaluate
            environment: Military environment profile
            num_samples: Number of samples to collect
            
        Returns:
            Dictionary containing sensitivity metrics
        """
        stressor = MilitaryStressors(environment=environment)
        base_response = self.puf.evaluate(challenge)
        
        times = np.random.uniform(0, 1000, num_samples)
        responses = []
        conditions = []
        
        for t in times:
            stressors = stressor.get_all_stressors(t)
            # Update PUF environmental conditions
            self.puf.environmental_stressors.update({
                'temperature': stressors['temperature'],
                'em_noise': stressors['em_noise'],
                'aging_factor': stressors['aging_factor']
            })
            response = self.puf.evaluate(challenge)
            responses.append(response)
            conditions.append(stressors)
            
        return {
            'times': times,
            'responses': np.array(responses),
            'conditions': conditions,
            'base_response': base_response
        }
        
    def plot_environmental_sensitivity(
        self,
        sensitivity_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """Create environmental sensitivity visualization.
        
        Args:
            sensitivity_data: Data from analyze_environmental_sensitivity
            save_path: Optional path to save the plot
        """
        responses = sensitivity_data['responses']
        times = sensitivity_data['times']
        conditions = sensitivity_data['conditions']
        
        # Extract condition data
        temps = np.array([c['temperature'] for c in conditions])
        emi = np.array([c['em_noise'] for c in conditions])
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # 3D scatter plot of response changes
        ax1 = fig.add_subplot(221, projection='3d')
        sc = ax1.scatter(temps, emi, times, c=responses, cmap='coolwarm')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('EMI Level')
        ax1.set_zlabel('Mission Time (h)')
        plt.colorbar(sc, label='Response')
        
        # Temperature sensitivity
        ax2 = fig.add_subplot(222)
        temp_bins = np.linspace(min(temps), max(temps), 20)
        temp_responses = [np.mean(responses[np.digitize(temps, temp_bins) == i])
                         for i in range(len(temp_bins))]
        ax2.plot(temp_bins, temp_responses, 'b-')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Response Probability')
        ax2.set_title('Temperature Sensitivity')
        ax2.grid(True)
        
        # EMI sensitivity
        ax3 = fig.add_subplot(223)
        emi_bins = np.linspace(min(emi), max(emi), 20)
        emi_responses = [np.mean(responses[np.digitize(emi, emi_bins) == i])
                        for i in range(len(emi_bins))]
        ax3.plot(emi_bins, emi_responses, 'r-')
        ax3.set_xlabel('EMI Level')
        ax3.set_ylabel('Response Probability')
        ax3.set_title('EMI Sensitivity')
        ax3.grid(True)
        
        # Time evolution
        ax4 = fig.add_subplot(224)
        time_bins = np.linspace(min(times), max(times), 20)
        time_responses = [np.mean(responses[np.digitize(times, time_bins) == i])
                         for i in range(len(time_bins))]
        ax4.plot(time_bins, time_responses, 'g-')
        ax4.set_xlabel('Mission Time (h)')
        ax4.set_ylabel('Response Probability')
        ax4.set_title('Time Evolution')
        ax4.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_reliability_report(
        self,
        challenge: np.ndarray,
        environment: MilitaryEnvironment,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Generate comprehensive reliability report for military applications.
        
        Args:
            challenge: Challenge bits to evaluate
            environment: Military environment profile
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary containing reliability metrics
        """
        # Analyze reliability under stress
        stress_data = self.analyze_reliability_under_stress(challenge)
        reliability_data = self.analyze_environmental_sensitivity(challenge, environment)
        
        # Calculate metrics
        mean_reliability = np.mean(stress_data['responses'] == stress_data['base_response']) * 100
        worst_case_reliability = np.min(np.mean(stress_data['responses'] == stress_data['base_response'], axis=1)) * 100
        temp_sensitivity = np.std([np.mean(stress_data['responses'][i] == stress_data['base_response'])
                                 for i in range(len(stress_data['temperatures']))]) * 100
        emi_sensitivity = np.std([np.mean(stress_data['responses'][i] == stress_data['base_response'])
                                for i in range(len(stress_data['emi_levels']))]) * 100
        
        # Generate visualizations if save path provided
        if save_path:
            self.plot_reliability_analysis(stress_data, f"{save_path}_reliability.png")
            self.plot_environmental_sensitivity(reliability_data, f"{save_path}_sensitivity.png")
        
        return {
            'mean_reliability_percent': mean_reliability,
            'worst_case_reliability_percent': worst_case_reliability,
            'temperature_sensitivity_percent': temp_sensitivity,
            'emi_sensitivity_percent': emi_sensitivity,
            'aging_impact': np.mean(stress_data['aging_factors']) - 1.0
        } 