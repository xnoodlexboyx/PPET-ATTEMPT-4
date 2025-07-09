"""PUF analysis and visualization module with military-grade capabilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

    def analyze_uniqueness(self, pufs: List[PUF], num_crps: int = 1000) -> Dict[str, np.ndarray]:
        """Analyze uniqueness between multiple PUF instances.
        
        Args:
            pufs: List of PUF instances to analyze
            num_crps: Number of challenge-response pairs to generate
            
        Returns:
            Dictionary containing uniqueness metrics and pairwise distances
        """
        num_pufs = len(pufs)
        responses = []
        
        # Generate same challenges for all PUFs
        if hasattr(pufs[0], 'n_stages'):
            # For ArbiterPUF
            challenges = np.random.randint(0, 2, size=(num_crps, pufs[0].n_stages))
        else:
            # For other PUF types, use generate_crps method
            challenges = None
            
        # Collect responses from all PUFs
        for puf in pufs:
            if challenges is not None:
                _, puf_responses = puf.generate_crps(num_crps, challenges)
            else:
                puf_responses = puf.generate_crps(num_crps)
                if isinstance(puf_responses, tuple):
                    puf_responses = puf_responses[1]
            responses.append(puf_responses)
        
        responses = np.array(responses)
        
        # Calculate pairwise Hamming distances
        hamming_distances = np.zeros((num_pufs, num_pufs))
        for i in range(num_pufs):
            for j in range(i+1, num_pufs):
                if responses.ndim == 3:  # SRAM PUF case
                    hamming_dist = np.mean(responses[i] != responses[j])
                else:  # Arbiter/RO PUF case
                    hamming_dist = np.mean(responses[i] != responses[j])
                hamming_distances[i, j] = hamming_distances[j, i] = hamming_dist
        
        # Calculate uniqueness metrics
        upper_triangle = np.triu_indices(num_pufs, k=1)
        pairwise_distances = hamming_distances[upper_triangle]
        
        return {
            'hamming_distances': hamming_distances,
            'pairwise_distances': pairwise_distances,
            'mean_uniqueness': np.mean(pairwise_distances),
            'std_uniqueness': np.std(pairwise_distances),
            'responses': responses,
            'num_pufs': num_pufs
        }

    def plot_uniqueness_analysis(self, uniqueness_data: Dict[str, np.ndarray], 
                               save_path: Optional[str] = None, use_plotly: bool = False):
        """Create comprehensive uniqueness visualization.
        
        Args:
            uniqueness_data: Data from analyze_uniqueness
            save_path: Optional path to save the plot
            use_plotly: Whether to use Plotly for interactive plots
        """
        if use_plotly:
            self._plot_uniqueness_plotly(uniqueness_data, save_path)
        else:
            self._plot_uniqueness_matplotlib(uniqueness_data, save_path)

    def _plot_uniqueness_matplotlib(self, uniqueness_data: Dict[str, np.ndarray], 
                                   save_path: Optional[str] = None):
        """Create uniqueness visualization using Matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram of pairwise distances
        axes[0, 0].hist(uniqueness_data['pairwise_distances'], bins=30, alpha=0.7, 
                       color='blue', edgecolor='black')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Ideal (50%)')
        axes[0, 0].axvline(uniqueness_data['mean_uniqueness'], color='green', 
                          linestyle='-', label=f'Mean ({uniqueness_data["mean_uniqueness"]:.3f})')
        axes[0, 0].set_xlabel('Hamming Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Pairwise Hamming Distances')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Heatmap of distance matrix
        im = axes[0, 1].imshow(uniqueness_data['hamming_distances'], cmap='viridis')
        axes[0, 1].set_title('Pairwise Hamming Distance Matrix')
        axes[0, 1].set_xlabel('PUF Index')
        axes[0, 1].set_ylabel('PUF Index')
        plt.colorbar(im, ax=axes[0, 1], label='Hamming Distance')
        
        # Scatter plot of distance vs PUF pair indices
        num_pufs = uniqueness_data['num_pufs']
        pair_indices = []
        distances = []
        for i in range(num_pufs):
            for j in range(i+1, num_pufs):
                pair_indices.append(f"{i}-{j}")
                distances.append(uniqueness_data['hamming_distances'][i, j])
        
        axes[1, 0].scatter(range(len(distances)), distances, alpha=0.6)
        axes[1, 0].axhline(0.5, color='red', linestyle='--', label='Ideal (50%)')
        axes[1, 0].set_xlabel('PUF Pair Index')
        axes[1, 0].set_ylabel('Hamming Distance')
        axes[1, 0].set_title('Uniqueness Scatter Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot of uniqueness distribution
        axes[1, 1].boxplot(uniqueness_data['pairwise_distances'], patch_artist=True)
        axes[1, 1].axhline(0.5, color='red', linestyle='--', label='Ideal (50%)')
        axes[1, 1].set_ylabel('Hamming Distance')
        axes[1, 1].set_title('Uniqueness Distribution (Box Plot)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_uniqueness_plotly(self, uniqueness_data: Dict[str, np.ndarray], 
                               save_path: Optional[str] = None):
        """Create interactive uniqueness visualization using Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hamming Distance Distribution', 'Distance Matrix Heatmap',
                           'Uniqueness Scatter Plot', 'Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=uniqueness_data['pairwise_distances'], name='Distances',
                        nbinsx=30, opacity=0.7),
            row=1, col=1
        )
        
        # Heatmap
        fig.add_trace(
            go.Heatmap(z=uniqueness_data['hamming_distances'], 
                      colorscale='viridis', name='Distance Matrix'),
            row=1, col=2
        )
        
        # Scatter plot
        num_pufs = uniqueness_data['num_pufs']
        distances = []
        for i in range(num_pufs):
            for j in range(i+1, num_pufs):
                distances.append(uniqueness_data['hamming_distances'][i, j])
        
        fig.add_trace(
            go.Scatter(x=list(range(len(distances))), y=distances, 
                      mode='markers', name='Pairwise Distances'),
            row=2, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=uniqueness_data['pairwise_distances'], name='Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="PUF Uniqueness Analysis")
        
        if save_path:
            fig.write_html(save_path)
        fig.show()

    def analyze_bit_aliasing(self, pufs: List[PUF], num_crps: int = 1000) -> Dict[str, np.ndarray]:
        """Analyze bit-aliasing patterns across PUF instances.
        
        Args:
            pufs: List of PUF instances to analyze
            num_crps: Number of challenge-response pairs to generate
            
        Returns:
            Dictionary containing bit-aliasing metrics
        """
        responses = []
        
        # Generate responses from all PUFs
        for puf in pufs:
            if hasattr(puf, 'n_stages'):
                # For ArbiterPUF - single bit responses
                challenges = np.random.randint(0, 2, size=(num_crps, puf.n_stages))
                _, puf_responses = puf.generate_crps(num_crps, challenges)
                responses.append(puf_responses)
            else:
                # For SRAM PUF - multi-bit responses
                puf_responses = puf.generate_crps(num_crps)
                if isinstance(puf_responses, tuple):
                    puf_responses = puf_responses[1]
                responses.append(puf_responses)
        
        responses = np.array(responses)
        
        # Calculate bit frequencies
        if responses.ndim == 2:  # Single bit responses (Arbiter/RO PUF)
            bit_frequencies = np.mean(responses, axis=0)
            bit_positions = np.arange(len(bit_frequencies))
        else:  # Multi-bit responses (SRAM PUF)
            # Flatten the responses and calculate frequencies per bit position
            flattened = responses.reshape(len(pufs), -1)
            bit_frequencies = np.mean(flattened, axis=0)
            bit_positions = np.arange(len(bit_frequencies))
        
        # Calculate aliasing metrics
        ideal_frequency = 0.5
        aliasing_deviation = np.abs(bit_frequencies - ideal_frequency)
        
        return {
            'bit_frequencies': bit_frequencies,
            'bit_positions': bit_positions,
            'aliasing_deviation': aliasing_deviation,
            'mean_aliasing': np.mean(aliasing_deviation),
            'max_aliasing': np.max(aliasing_deviation),
            'responses': responses
        }

    def plot_bit_aliasing_analysis(self, aliasing_data: Dict[str, np.ndarray], 
                                  save_path: Optional[str] = None, use_plotly: bool = False):
        """Create comprehensive bit-aliasing visualization.
        
        Args:
            aliasing_data: Data from analyze_bit_aliasing
            save_path: Optional path to save the plot
            use_plotly: Whether to use Plotly for interactive plots
        """
        if use_plotly:
            self._plot_bit_aliasing_plotly(aliasing_data, save_path)
        else:
            self._plot_bit_aliasing_matplotlib(aliasing_data, save_path)

    def _plot_bit_aliasing_matplotlib(self, aliasing_data: Dict[str, np.ndarray], 
                                     save_path: Optional[str] = None):
        """Create bit-aliasing visualization using Matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar graph of bit frequencies
        axes[0, 0].bar(aliasing_data['bit_positions'], aliasing_data['bit_frequencies'], 
                       alpha=0.7, color='blue')
        axes[0, 0].axhline(0.5, color='red', linestyle='--', label='Ideal (50%)')
        axes[0, 0].set_xlabel('Bit Position')
        axes[0, 0].set_ylabel('Frequency of 1s')
        axes[0, 0].set_title('Bit Frequencies Across Positions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Heatmap of bit frequencies (reshaped if possible)
        if len(aliasing_data['bit_frequencies']) > 1:
            # Try to reshape into a 2D grid for better visualization
            n_bits = len(aliasing_data['bit_frequencies'])
            grid_size = int(np.sqrt(n_bits))
            if grid_size * grid_size == n_bits:
                freq_matrix = aliasing_data['bit_frequencies'].reshape(grid_size, grid_size)
            else:
                # Create a rectangular grid
                rows = int(np.sqrt(n_bits))
                cols = int(np.ceil(n_bits / rows))
                padded_freq = np.pad(aliasing_data['bit_frequencies'], 
                                   (0, rows * cols - n_bits), mode='constant', constant_values=0.5)
                freq_matrix = padded_freq.reshape(rows, cols)
            
            im = axes[0, 1].imshow(freq_matrix, cmap='RdBu_r', vmin=0, vmax=1)
            axes[0, 1].set_title('Bit Frequency Heatmap')
            axes[0, 1].set_xlabel('Column')
            axes[0, 1].set_ylabel('Row')
            plt.colorbar(im, ax=axes[0, 1], label='Frequency of 1s')
        
        # Aliasing deviation plot
        axes[1, 0].plot(aliasing_data['bit_positions'], aliasing_data['aliasing_deviation'], 
                       'o-', alpha=0.7, color='green')
        axes[1, 0].axhline(aliasing_data['mean_aliasing'], color='red', linestyle='--', 
                          label=f'Mean ({aliasing_data["mean_aliasing"]:.3f})')
        axes[1, 0].set_xlabel('Bit Position')
        axes[1, 0].set_ylabel('Aliasing Deviation')
        axes[1, 0].set_title('Bit Aliasing Deviation from Ideal')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of aliasing deviations
        axes[1, 1].hist(aliasing_data['aliasing_deviation'], bins=30, alpha=0.7, 
                       color='orange', edgecolor='black')
        axes[1, 1].axvline(aliasing_data['mean_aliasing'], color='red', linestyle='--', 
                          label=f'Mean ({aliasing_data["mean_aliasing"]:.3f})')
        axes[1, 1].set_xlabel('Aliasing Deviation')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Aliasing Deviations')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_bit_aliasing_plotly(self, aliasing_data: Dict[str, np.ndarray], 
                                 save_path: Optional[str] = None):
        """Create interactive bit-aliasing visualization using Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bit Frequencies', 'Frequency Heatmap',
                           'Aliasing Deviation', 'Deviation Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bar graph
        fig.add_trace(
            go.Bar(x=aliasing_data['bit_positions'], y=aliasing_data['bit_frequencies'], 
                   name='Bit Frequencies'),
            row=1, col=1
        )
        
        # Heatmap (if possible to reshape)
        n_bits = len(aliasing_data['bit_frequencies'])
        grid_size = int(np.sqrt(n_bits))
        if grid_size * grid_size == n_bits:
            freq_matrix = aliasing_data['bit_frequencies'].reshape(grid_size, grid_size)
            fig.add_trace(
                go.Heatmap(z=freq_matrix, colorscale='RdBu_r', name='Frequency Heatmap'),
                row=1, col=2
            )
        
        # Line plot of deviations
        fig.add_trace(
            go.Scatter(x=aliasing_data['bit_positions'], y=aliasing_data['aliasing_deviation'],
                      mode='lines+markers', name='Aliasing Deviation'),
            row=2, col=1
        )
        
        # Histogram of deviations
        fig.add_trace(
            go.Histogram(x=aliasing_data['aliasing_deviation'], name='Deviation Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="PUF Bit-Aliasing Analysis")
        
        if save_path:
            fig.write_html(save_path)
        fig.show()

    def generate_comprehensive_report(self, pufs: List[PUF], environment: MilitaryEnvironment,
                                    num_crps: int = 1000, save_dir: Optional[str] = None,
                                    use_plotly: bool = False) -> Dict[str, float]:
        """Generate comprehensive PUF analysis report with all visualizations.
        
        Args:
            pufs: List of PUF instances to analyze
            environment: Military environment profile
            num_crps: Number of challenge-response pairs to generate
            save_dir: Directory to save visualizations
            use_plotly: Whether to use Plotly for interactive plots
            
        Returns:
            Dictionary containing all analysis metrics
        """
        # Initialize results dictionary
        results = {}
        
        # Analyze single PUF reliability under stress
        if pufs:
            reliability_data = self.analyze_reliability_under_stress(
                np.random.randint(0, 2, size=getattr(pufs[0], 'n_stages', 64))
            )
            results.update(self.generate_reliability_report(
                np.random.randint(0, 2, size=getattr(pufs[0], 'n_stages', 64)),
                environment,
                f"{save_dir}/reliability" if save_dir else None
            ))
            
            # Analyze uniqueness across PUFs
            uniqueness_data = self.analyze_uniqueness(pufs, num_crps)
            results.update({
                'mean_uniqueness': uniqueness_data['mean_uniqueness'],
                'std_uniqueness': uniqueness_data['std_uniqueness']
            })
            
            # Analyze bit-aliasing
            aliasing_data = self.analyze_bit_aliasing(pufs, num_crps)
            results.update({
                'mean_aliasing': aliasing_data['mean_aliasing'],
                'max_aliasing': aliasing_data['max_aliasing']
            })
            
            # Generate visualizations
            if save_dir:
                self.plot_uniqueness_analysis(uniqueness_data, 
                                            f"{save_dir}/uniqueness.{'html' if use_plotly else 'png'}", 
                                            use_plotly)
                self.plot_bit_aliasing_analysis(aliasing_data, 
                                               f"{save_dir}/bit_aliasing.{'html' if use_plotly else 'png'}", 
                                               use_plotly)
        
        return results 