#!/usr/bin/env python3
"""
Comprehensive Threat Assessment Example

This script demonstrates the complete threat assessment report generation
capabilities for defense procurement and security evaluation.
"""

import numpy as np
import sys
import os
sys.path.append('.')

from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF
from ppet.core.military_stressors import Military
Environment
from ppet.core.threat_assessment import ThreatAssessmentReportGenerator

def create_defense_puf_suite() -> list:
    """Create comprehensive PUF suite for defense evaluation."""
    pufs = []
    
    # Create diverse PUF types for comprehensive evaluation
    for i in range(15):
        if i < 5:
            # Arbiter PUFs - various sizes for different applications
            puf = ArbiterPUF(n_stages=64 + i*16, seed=i)
        elif i < 10:
            # SRAM PUFs - different array sizes
            size = 16 + (i-5)*8
            puf = SRAMPUF(rows=size, columns=size, seed=i)
        else:
            # Ring Oscillator PUFs - various oscillator counts
            puf = RingOscillatorPUF(num_oscillators=100 + (i-10)*50, seed=i)
        
        pufs.append(puf)
    
    return pufs

def demonstrate_ground_mobile_assessment():
    """Demonstrate ground mobile system assessment."""
    print("=== Ground Mobile System Assessment ===")
    
    # Create PUF suite
    pufs = create_defense_puf_suite()
    
    # Create report generator
    report_gen = ThreatAssessmentReportGenerator(classification_level="CONFIDENTIAL")
    
    # Generate comprehensive report
    print("Generating comprehensive threat assessment report...")
    report_data = report_gen.generate_defense_procurement_report(
        pufs=pufs,
        environment=MilitaryEnvironment.GROUND_MOBILE,
        use_cases=["Tank Authentication", "Secure Radio Communication", "Battlefield IoT"],
        output_dir="ground_mobile_threat_assessment"
    )
    
    # Display summary
    print("\n=== ASSESSMENT SUMMARY ===")
    print(f"Overall Risk Level: {report_data['assessment_summary']['risk_level']}")
    print(f"Risk Score: {report_data['assessment_summary']['overall_risk_score']:.2f}")
    print(f"Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}")
    print(f"Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%")
    
    return report_data

def demonstrate_aircraft_assessment():
    """Demonstrate aircraft system assessment."""
    print("\n=== Aircraft System Assessment ===")
    
    # Create PUF suite optimized for aircraft
    pufs = []
    for i in range(10):
        # Use more stages for aircraft applications (higher reliability needed)
        puf = ArbiterPUF(n_stages=128, seed=i + 100)
        # Set aircraft-specific environmental conditions
        puf.environmental_stressors = {
            'temperature': -30.0,  # High altitude cold
            'voltage': 1.15,       # Aircraft power systems
            'em_noise': 0.4,       # Aircraft electronics
            'aging_factor': 1.0
        }
        pufs.append(puf)
    
    # Create report generator
    report_gen = ThreatAssessmentReportGenerator(classification_level="SECRET")
    
    # Generate report
    print("Generating aircraft threat assessment report...")
    report_data = report_gen.generate_defense_procurement_report(
        pufs=pufs,
        environment=MilitaryEnvironment.AIRCRAFT_INTERNAL,
        use_cases=["Avionics Authentication", "Secure Data Link", "Mission Computer Security"],
        output_dir="aircraft_threat_assessment"
    )
    
    # Display summary
    print("\n=== AIRCRAFT ASSESSMENT SUMMARY ===")
    print(f"Overall Risk Level: {report_data['assessment_summary']['risk_level']}")
    print(f"Risk Score: {report_data['assessment_summary']['overall_risk_score']:.2f}")
    print(f"Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}")
    print(f"Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%")
    
    return report_data

def demonstrate_space_vehicle_assessment():
    """Demonstrate space vehicle system assessment."""
    print("\n=== Space Vehicle System Assessment ===")
    
    # Create PUF suite for space applications
    pufs = []
    for i in range(8):
        # Use maximum stages for space reliability
        puf = ArbiterPUF(n_stages=256, seed=i + 200)
        # Set space-specific environmental conditions
        puf.environmental_stressors = {
            'temperature': -60.0,  # Space cold
            'voltage': 1.25,       # Space-qualified power
            'em_noise': 0.8,       # High radiation environment
            'aging_factor': 1.3    # Accelerated aging in space
        }
        pufs.append(puf)
    
    # Create report generator
    report_gen = ThreatAssessmentReportGenerator(classification_level="TOP_SECRET")
    
    # Generate report
    print("Generating space vehicle threat assessment report...")
    report_data = report_gen.generate_defense_procurement_report(
        pufs=pufs,
        environment=MilitaryEnvironment.SPACE_VEHICLE,
        use_cases=["Satellite Authentication", "Secure Telemetry", "Command Authorization"],
        output_dir="space_vehicle_threat_assessment"
    )
    
    # Display summary
    print("\n=== SPACE VEHICLE ASSESSMENT SUMMARY ===")
    print(f"Overall Risk Level: {report_data['assessment_summary']['risk_level']}")
    print(f"Risk Score: {report_data['assessment_summary']['overall_risk_score']:.2f}")
    print(f"Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}")
    print(f"Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%")
    
    return report_data

def demonstrate_naval_assessment():
    """Demonstrate naval system assessment."""
    print("\n=== Naval System Assessment ===")
    
    # Create PUF suite for naval applications
    pufs = []
    for i in range(12):
        if i < 6:
            # SRAM PUFs for naval applications
            puf = SRAMPUF(rows=32, columns=32, seed=i + 300)
        else:
            # Ring Oscillator PUFs for naval applications
            puf = RingOscillatorPUF(num_oscillators=150, seed=i + 300)
        
        # Set naval-specific environmental conditions
        puf.environmental_stressors = {
            'temperature': 35.0,   # Warm ocean environment
            'voltage': 1.2,        # Naval power systems
            'em_noise': 0.5,       # Naval electronics
            'aging_factor': 1.1    # Marine corrosion effects
        }
        pufs.append(puf)
    
    # Create report generator
    report_gen = ThreatAssessmentReportGenerator(classification_level="SECRET")
    
    # Generate report
    print("Generating naval threat assessment report...")
    report_data = report_gen.generate_defense_procurement_report(
        pufs=pufs,
        environment=MilitaryEnvironment.NAVAL_EXPOSED,
        use_cases=["Ship Authentication", "Secure Navigation", "Sonar Data Protection"],
        output_dir="naval_threat_assessment"
    )
    
    # Display summary
    print("\n=== NAVAL ASSESSMENT SUMMARY ===")
    print(f"Overall Risk Level: {report_data['assessment_summary']['risk_level']}")
    print(f"Risk Score: {report_data['assessment_summary']['overall_risk_score']:.2f}")
    print(f"Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}")
    print(f"Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%")
    
    return report_data

def generate_comparative_analysis(all_reports: list):
    """Generate comparative analysis across all environments."""
    print("\n=== COMPARATIVE ANALYSIS ===")
    
    environments = ['Ground Mobile', 'Aircraft', 'Space Vehicle', 'Naval']
    
    print("Risk Level Comparison:")
    for i, env in enumerate(environments):
        risk_level = all_reports[i]['assessment_summary']['risk_level']
        risk_score = all_reports[i]['assessment_summary']['overall_risk_score']
        print(f"  {env}: {risk_level} (Score: {risk_score:.2f})")
    
    print("\nUniqueness Comparison:")
    for i, env in enumerate(environments):
        uniqueness = all_reports[i]['assessment_summary']['uniqueness_score']
        print(f"  {env}: {uniqueness:.3f}")
    
    print("\nReliability Comparison:")
    for i, env in enumerate(environments):
        reliability = all_reports[i]['assessment_summary']['mean_reliability_percent']
        print(f"  {env}: {reliability:.1f}%")
    
    # Find best and worst performing environments
    risk_scores = [r['assessment_summary']['overall_risk_score'] for r in all_reports]
    best_env = environments[np.argmin(risk_scores)]
    worst_env = environments[np.argmax(risk_scores)]
    
    print(f"\nBest Performing Environment: {best_env}")
    print(f"Worst Performing Environment: {worst_env}")
    
    return {
        'best_environment': best_env,
        'worst_environment': worst_env,
        'risk_scores': dict(zip(environments, risk_scores))
    }

def main():
    """Main demonstration function."""
    print("PPET Comprehensive Threat Assessment Demonstration")
    print("=" * 60)
    
    try:
        # Run all assessments
        reports = []
        
        # Ground Mobile Assessment
        ground_report = demonstrate_ground_mobile_assessment()
        reports.append(ground_report)
        
        # Aircraft Assessment
        aircraft_report = demonstrate_aircraft_assessment()
        reports.append(aircraft_report)
        
        # Space Vehicle Assessment
        space_report = demonstrate_space_vehicle_assessment()
        reports.append(space_report)
        
        # Naval Assessment
        naval_report = demonstrate_naval_assessment()
        reports.append(naval_report)
        
        # Comparative Analysis
        comparative_analysis = generate_comparative_analysis(reports)
        
        print("\n" + "=" * 60)
        print("All threat assessments completed successfully!")
        print("\nGenerated Report Directories:")
        print("- ground_mobile_threat_assessment/")
        print("- aircraft_threat_assessment/")
        print("- space_vehicle_threat_assessment/")
        print("- naval_threat_assessment/")
        
        print("\nEach directory contains:")
        print("- threat_assessment_report.html (Main report)")
        print("- threat_assessment_report.json (Detailed data)")
        print("- Multiple 3D visualization files (.html)")
        print("- Reliability analysis files (.png)")
        
        print("\nOpen the HTML files in a web browser to view the reports!")
        
    except Exception as e:
        print(f"Error during assessment: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()