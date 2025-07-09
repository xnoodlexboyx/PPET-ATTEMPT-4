#!/usr/bin/env python3
"""
Comprehensive PUF Analysis Example

This script demonstrates the enhanced analysis capabilities of the PPET framework,
including uniqueness analysis, bit-aliasing analysis, and comprehensive reporting
with both Matplotlib and Plotly visualizations.
"""

import numpy as np
import os
from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF
from ppet.core.analysis import PUFAnalyzer
from ppet.core.military_stressors import MilitaryEnvironment

def create_sample_pufs(num_pufs: int = 10, puf_type: str = 'arbiter'):
    """Create a list of sample PUFs for analysis.
    
    Args:
        num_pufs: Number of PUF instances to create
        puf_type: Type of PUF ('arbiter', 'sram', 'ring_oscillator')
    
    Returns:
        List of PUF instances
    """
    pufs = []
    
    for i in range(num_pufs):
        if puf_type == 'arbiter':
            puf = ArbiterPUF(n_stages=64, seed=i)
        elif puf_type == 'sram':
            puf = SRAMPUF(rows=16, columns=16, seed=i)
        elif puf_type == 'ring_oscillator':
            puf = RingOscillatorPUF(num_oscillators=100, seed=i)
        else:
            raise ValueError(f"Unknown PUF type: {puf_type}")
        
        pufs.append(puf)
    
    return pufs

def demonstrate_uniqueness_analysis():
    """Demonstrate uniqueness analysis with visualizations."""
    print("=== Uniqueness Analysis Demo ===")
    
    # Create multiple Arbiter PUFs
    pufs = create_sample_pufs(num_pufs=15, puf_type='arbiter')
    analyzer = PUFAnalyzer(pufs[0])
    
    # Analyze uniqueness
    uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=1000)
    
    # Display results
    print(f"Mean uniqueness: {uniqueness_data['mean_uniqueness']:.3f}")
    print(f"Standard deviation: {uniqueness_data['std_uniqueness']:.3f}")
    print(f"Number of PUF pairs: {len(uniqueness_data['pairwise_distances'])}")
    
    # Create visualizations
    print("Creating Matplotlib visualization...")
    analyzer.plot_uniqueness_analysis(uniqueness_data, 
                                     save_path='uniqueness_matplotlib.png', 
                                     use_plotly=False)
    
    print("Creating Plotly visualization...")
    analyzer.plot_uniqueness_analysis(uniqueness_data, 
                                     save_path='uniqueness_plotly.html', 
                                     use_plotly=True)
    
    return uniqueness_data

def demonstrate_bit_aliasing_analysis():
    """Demonstrate bit-aliasing analysis with visualizations."""
    print("\n=== Bit-Aliasing Analysis Demo ===")
    
    # Create multiple SRAM PUFs
    pufs = create_sample_pufs(num_pufs=20, puf_type='sram')
    analyzer = PUFAnalyzer(pufs[0])
    
    # Analyze bit-aliasing
    aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=500)
    
    # Display results
    print(f"Mean aliasing deviation: {aliasing_data['mean_aliasing']:.3f}")
    print(f"Maximum aliasing deviation: {aliasing_data['max_aliasing']:.3f}")
    print(f"Number of bit positions: {len(aliasing_data['bit_positions'])}")
    
    # Create visualizations
    print("Creating Matplotlib visualization...")
    analyzer.plot_bit_aliasing_analysis(aliasing_data, 
                                       save_path='bit_aliasing_matplotlib.png', 
                                       use_plotly=False)
    
    print("Creating Plotly visualization...")
    analyzer.plot_bit_aliasing_analysis(aliasing_data, 
                                       save_path='bit_aliasing_plotly.html', 
                                       use_plotly=True)
    
    return aliasing_data

def demonstrate_comprehensive_report():
    """Demonstrate comprehensive report generation."""
    print("\n=== Comprehensive Report Demo ===")
    
    # Create multiple PUFs
    pufs = create_sample_pufs(num_pufs=12, puf_type='arbiter')
    analyzer = PUFAnalyzer(pufs[0])
    
    # Define military environment
    environment = MilitaryEnvironment.DESERT_OPERATION
    
    # Create output directory
    output_dir = "comprehensive_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    print("Generating comprehensive analysis report...")
    results = analyzer.generate_comprehensive_report(
        pufs=pufs,
        environment=environment,
        num_crps=1000,
        save_dir=output_dir,
        use_plotly=False  # Use Matplotlib for this demo
    )
    
    # Display summary results
    print("\n=== Analysis Summary ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nVisualization files saved to: {output_dir}/")
    return results

def demonstrate_environmental_stress_analysis():
    """Demonstrate environmental stress analysis."""
    print("\n=== Environmental Stress Analysis Demo ===")
    
    # Create a PUF for stress testing
    puf = ArbiterPUF(n_stages=64, seed=42)
    analyzer = PUFAnalyzer(puf)
    
    # Define test challenge
    challenge = np.random.randint(0, 2, size=64)
    
    # Analyze reliability under stress
    print("Analyzing reliability under military stress conditions...")
    stress_data = analyzer.analyze_reliability_under_stress(
        challenge=challenge,
        num_trials=50,
        time_points=np.linspace(0, 1000, 25)  # 25 time points over 1000 hours
    )
    
    # Create visualization
    analyzer.plot_reliability_analysis(stress_data, 
                                     save_path='stress_analysis.png')
    
    # Display results
    reliability_over_time = np.mean(stress_data['responses'] == stress_data['base_response'], axis=1)
    print(f"Initial reliability: {reliability_over_time[0]:.3f}")
    print(f"Final reliability: {reliability_over_time[-1]:.3f}")
    print(f"Mean reliability: {np.mean(reliability_over_time):.3f}")
    print(f"Temperature range: {np.min(stress_data['temperatures']):.1f}°C to {np.max(stress_data['temperatures']):.1f}°C")
    
    return stress_data

def main():
    """Main demonstration function."""
    print("PPET Comprehensive Analysis Demonstration")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        uniqueness_data = demonstrate_uniqueness_analysis()
        aliasing_data = demonstrate_bit_aliasing_analysis()
        comprehensive_results = demonstrate_comprehensive_report()
        stress_data = demonstrate_environmental_stress_analysis()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nGenerated files:")
        print("- uniqueness_matplotlib.png")
        print("- uniqueness_plotly.html")
        print("- bit_aliasing_matplotlib.png")
        print("- bit_aliasing_plotly.html")
        print("- comprehensive_analysis_results/ (directory)")
        print("- stress_analysis.png")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()