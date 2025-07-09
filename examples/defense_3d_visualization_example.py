#!/usr/bin/env python3
"""
Defense-Oriented 3D Visualization Example

This script demonstrates comprehensive 3D visualization capabilities for defense scenarios,
including threat landscape analysis, environmental stress impact, and multi-attack comparisons.
"""

import numpy as np
import os
from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF
from ppet.core.analysis import PUFAnalyzer
from ppet.core.military_stressors import MilitaryEnvironment
from ppet.core.threat_simulator import MLAttack, SideChannelAttack, SupplyChainAttack, FaultInjectionAttack

def create_defense_pufs(num_pufs: int = 5) -> list:
    """Create a collection of defense-oriented PUFs."""
    pufs = []
    
    # Create diverse PUF types for comprehensive analysis
    for i in range(num_pufs):
        if i % 3 == 0:
            puf = ArbiterPUF(n_stages=128, seed=i)  # Larger challenge for defense
        elif i % 3 == 1:
            puf = SRAMPUF(rows=32, columns=32, seed=i)  # Larger SRAM for defense
        else:
            puf = RingOscillatorPUF(num_oscillators=200, seed=i)  # More oscillators
        
        pufs.append(puf)
    
    return pufs

def create_defense_attacks() -> list:
    """Create a collection of defense-relevant attacks."""
    attacks = []
    
    # Machine Learning Attacks
    attacks.append(MLAttack(model_type='rf', environmental_augmentation=True, 
                           military_environment=MilitaryEnvironment.GROUND_MOBILE))
    attacks.append(MLAttack(model_type='mlp', environmental_augmentation=False))
    
    # Side-Channel Attacks
    attacks.append(SideChannelAttack(attack_type='power', noise_std=0.1))
    attacks.append(SideChannelAttack(attack_type='timing', noise_std=0.05))
    
    # Supply Chain Attack
    attacks.append(SupplyChainAttack(tampering_rate=0.02, detection_difficulty=0.9))
    
    # Fault Injection Attacks
    attacks.append(FaultInjectionAttack(injection_type='voltage', precision=0.8))
    attacks.append(FaultInjectionAttack(injection_type='laser', precision=0.9))
    
    return attacks

def demonstrate_3d_threat_landscape():
    """Demonstrate 3D threat landscape visualization."""
    print("=== 3D Threat Landscape Analysis ===")
    
    # Create PUFs and attacks
    pufs = create_defense_pufs(3)
    attacks = create_defense_attacks()[:4]  # Use first 4 attacks for clarity
    
    # Create analyzer
    analyzer = PUFAnalyzer(pufs[0])
    
    # Create 3D threat landscape
    print("Generating 3D threat landscape visualization...")
    analyzer.plot_3d_threat_landscape(
        pufs=pufs,
        attacks=attacks,
        environment=MilitaryEnvironment.GROUND_MOBILE,
        save_path='3d_threat_landscape.html'
    )
    
    print("3D threat landscape saved to: 3d_threat_landscape.html")

def demonstrate_3d_puf_response_analysis():
    """Demonstrate 3D PUF response analysis."""
    print("\n=== 3D PUF Response Analysis ===")
    
    # Create PUFs
    pufs = create_defense_pufs(8)
    
    # Create analyzer
    analyzer = PUFAnalyzer(pufs[0])
    
    # Create 3D response analysis
    print("Generating 3D PUF response analysis...")
    analyzer.plot_3d_puf_response_analysis(
        pufs=pufs,
        num_crps=1000,
        save_path='3d_puf_response_analysis.html'
    )
    
    print("3D PUF response analysis saved to: 3d_puf_response_analysis.html")

def demonstrate_3d_environmental_stress():
    """Demonstrate 3D environmental stress impact visualization."""
    print("\n=== 3D Environmental Stress Impact ===")
    
    # Test different military environments
    environments = [
        MilitaryEnvironment.GROUND_MOBILE,
        MilitaryEnvironment.AIRCRAFT_EXTERNAL,
        MilitaryEnvironment.SPACE_VEHICLE
    ]
    
    # Create PUF
    puf = ArbiterPUF(n_stages=128, seed=42)
    analyzer = PUFAnalyzer(puf)
    
    for env in environments:
        print(f"Analyzing {env.value.upper()} environment...")
        analyzer.plot_3d_environmental_stress_impact(
            puf=puf,
            environment=env,
            save_path=f'3d_environmental_stress_{env.value}.html'
        )
        print(f"3D environmental stress for {env.value} saved to: 3d_environmental_stress_{env.value}.html")

def demonstrate_3d_multi_attack_comparison():
    """Demonstrate 3D multi-attack comparison visualization."""
    print("\n=== 3D Multi-Attack Comparison ===")
    
    # Create PUFs and attacks
    pufs = create_defense_pufs(3)
    attacks = create_defense_attacks()
    
    # Create analyzer
    analyzer = PUFAnalyzer(pufs[0])
    
    # Create 3D multi-attack comparison
    print("Generating 3D multi-attack comparison...")
    analyzer.plot_3d_multi_attack_comparison(
        pufs=pufs,
        attacks=attacks,
        save_path='3d_multi_attack_comparison.html'
    )
    
    print("3D multi-attack comparison saved to: 3d_multi_attack_comparison.html")

def demonstrate_satellite_communication_analysis():
    """Demonstrate satellite communication stress testing."""
    print("\n=== Satellite Communication Analysis ===")
    
    # Create space-hardened PUF
    puf = ArbiterPUF(n_stages=256, seed=123)  # Larger for space applications
    
    # Set space environment
    puf.environmental_stressors = {
        'temperature': -65.0,  # Space cold
        'voltage': 1.2,
        'em_noise': 0.8,  # High radiation environment
        'aging_factor': 1.0
    }
    
    analyzer = PUFAnalyzer(puf)
    
    # Analyze space environment impact
    print("Analyzing space environment impact on satellite communication...")
    analyzer.plot_3d_environmental_stress_impact(
        puf=puf,
        environment=MilitaryEnvironment.SPACE_VEHICLE,
        save_path='satellite_communication_stress.html'
    )
    
    # Generate reliability report
    challenge = np.random.randint(0, 2, size=256)
    reliability_metrics = analyzer.generate_reliability_report(
        challenge=challenge,
        environment=MilitaryEnvironment.SPACE_VEHICLE,
        save_path='satellite_reliability'
    )
    
    print("Satellite Communication Reliability Metrics:")
    for key, value in reliability_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print("Satellite analysis saved to: satellite_communication_stress.html")

def demonstrate_battlefield_iot_analysis():
    """Demonstrate battlefield IoT device analysis."""
    print("\n=== Battlefield IoT Device Analysis ===")
    
    # Create battlefield-hardened PUFs
    iot_pufs = []
    for i in range(10):  # 10 IoT devices
        puf = ArbiterPUF(n_stages=64, seed=i)  # Smaller for IoT constraints
        # Set ground mobile environment
        puf.environmental_stressors = {
            'temperature': 45.0,  # Hot battlefield
            'voltage': 1.1,       # Lower voltage for power saving
            'em_noise': 0.6,      # Radio interference
            'aging_factor': 1.2   # Accelerated aging
        }
        iot_pufs.append(puf)
    
    analyzer = PUFAnalyzer(iot_pufs[0])
    
    # Analyze uniqueness across IoT devices
    print("Analyzing uniqueness across battlefield IoT devices...")
    uniqueness_data = analyzer.analyze_uniqueness(iot_pufs, num_crps=500)
    
    print("Battlefield IoT Uniqueness Metrics:")
    print(f"  Mean uniqueness: {uniqueness_data['mean_uniqueness']:.3f}")
    print(f"  Std uniqueness: {uniqueness_data['std_uniqueness']:.3f}")
    
    # Create visualization
    analyzer.plot_uniqueness_analysis(
        uniqueness_data, 
        save_path='battlefield_iot_uniqueness.html', 
        use_plotly=True
    )
    
    # 3D PUF response analysis for IoT devices
    analyzer.plot_3d_puf_response_analysis(
        pufs=iot_pufs,
        num_crps=500,
        save_path='battlefield_iot_3d_analysis.html'
    )
    
    print("Battlefield IoT analysis saved to: battlefield_iot_*.html")

def demonstrate_drone_swarm_authentication():
    """Demonstrate drone swarm authentication analysis."""
    print("\n=== Drone Swarm Authentication Analysis ===")
    
    # Create drone swarm PUFs
    drone_pufs = []
    for i in range(20):  # 20 drones in swarm
        puf = ArbiterPUF(n_stages=128, seed=i + 1000)
        # Set aircraft environment
        puf.environmental_stressors = {
            'temperature': 25.0,  # Controlled aircraft internal
            'voltage': 1.2,
            'em_noise': 0.3,      # Moderate interference
            'aging_factor': 1.1   # Slight aging
        }
        drone_pufs.append(puf)
    
    analyzer = PUFAnalyzer(drone_pufs[0])
    
    # Analyze drone authentication reliability
    print("Analyzing drone swarm authentication reliability...")
    
    # Test various attack scenarios
    attacks = [
        MLAttack(model_type='rf', environmental_augmentation=True, 
                military_environment=MilitaryEnvironment.AIRCRAFT_INTERNAL),
        SideChannelAttack(attack_type='power', noise_std=0.08),
        SupplyChainAttack(tampering_rate=0.01, detection_difficulty=0.95)
    ]
    
    # 3D threat landscape for drone swarm
    analyzer.plot_3d_threat_landscape(
        pufs=drone_pufs[:5],  # Use 5 drones for analysis
        attacks=attacks,
        environment=MilitaryEnvironment.AIRCRAFT_INTERNAL,
        save_path='drone_swarm_threat_landscape.html'
    )
    
    # 3D multi-attack comparison
    analyzer.plot_3d_multi_attack_comparison(
        pufs=drone_pufs[:3],
        attacks=attacks,
        save_path='drone_swarm_attack_comparison.html'
    )
    
    print("Drone swarm analysis saved to: drone_swarm_*.html")

def main():
    """Main demonstration function."""
    print("PPET Defense-Oriented 3D Visualization Demonstration")
    print("=" * 60)
    
    try:
        # Create output directory
        output_dir = "defense_3d_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        
        # Run all demonstrations
        demonstrate_3d_threat_landscape()
        demonstrate_3d_puf_response_analysis()
        demonstrate_3d_environmental_stress()
        demonstrate_3d_multi_attack_comparison()
        demonstrate_satellite_communication_analysis()
        demonstrate_battlefield_iot_analysis()
        demonstrate_drone_swarm_authentication()
        
        print("\n" + "=" * 60)
        print("All 3D visualizations completed successfully!")
        print(f"\nGenerated files in '{output_dir}/':")
        print("- 3d_threat_landscape.html")
        print("- 3d_puf_response_analysis.html")
        print("- 3d_environmental_stress_*.html (multiple environments)")
        print("- 3d_multi_attack_comparison.html")
        print("- satellite_communication_stress.html")
        print("- battlefield_iot_*.html (multiple files)")
        print("- drone_swarm_*.html (multiple files)")
        print("- Various reliability and analysis PNG files")
        
        print("\nOpen the HTML files in a web browser to view interactive 3D visualizations!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()