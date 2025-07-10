#!/usr/bin/env python3
"""
Comprehensive Visualization Test Script for PPET Framework

This script provides exhaustive testing of all visualization capabilities in the PPET framework,
including all PUF types, all visualization functions, and comprehensive validation.
"""

import os
import sys
import time
import json
import traceback
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF
from ppet.core.analysis import PUFAnalyzer
from ppet.core.military_stressors import MilitaryEnvironment
from ppet.core.threat_assessment import ThreatAssessmentReportGenerator
from ppet.core.threat_simulator import MLAttack, SideChannelAttack, SupplyChainAttack, FaultInjectionAttack

class ComprehensiveVisualizationTester:
    """Comprehensive tester for all PPET visualization capabilities."""
    
    def __init__(self, output_dir: str = "comprehensive_viz_test_results"):
        self.output_dir = output_dir
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'puf_tests': {},
            'visualization_tests': {},
            'performance_metrics': {},
            'error_log': [],
            'summary': {}
        }
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/puf_tests", exist_ok=True)
        os.makedirs(f"{output_dir}/visualization_tests", exist_ok=True)
        os.makedirs(f"{output_dir}/thesis_data", exist_ok=True)
        os.makedirs(f"{output_dir}/performance_logs", exist_ok=True)
        
        print(f"Comprehensive Visualization Test Suite")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        
    def create_test_pufs(self, puf_type: str, num_pufs: int = 10) -> List:
        """Create test PUF instances of specified type."""
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
    
    def create_test_attacks(self) -> List:
        """Create test attack instances."""
        attacks = []
        
        try:
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
            
        except Exception as e:
            print(f"Warning: Could not create all attack types: {e}")
        
        return attacks
    
    def test_puf_types(self):
        """Test all PUF emulator types."""
        print("\n=== Testing PUF Emulator Types ===")
        
        puf_types = ['arbiter', 'sram', 'ring_oscillator']
        
        for puf_type in puf_types:
            print(f"\nTesting {puf_type.upper()} PUF...")
            
            try:
                start_time = time.time()
                
                # Create test PUFs
                pufs = self.create_test_pufs(puf_type, num_pufs=5)
                
                # Test basic functionality
                test_results = {
                    'creation_successful': True,
                    'num_pufs_created': len(pufs),
                    'crp_generation_successful': False,
                    'sample_crps': None,
                    'performance_metrics': {}
                }
                
                # Test CRP generation
                if pufs:
                    if hasattr(pufs[0], 'generate_crps'):
                        sample_crps = pufs[0].generate_crps(10)
                        test_results['crp_generation_successful'] = True
                        test_results['sample_crps'] = str(type(sample_crps))
                
                # Performance metrics
                end_time = time.time()
                test_results['performance_metrics'] = {
                    'creation_time': end_time - start_time,
                    'memory_efficiency': 'good' if len(pufs) == 5 else 'poor'
                }
                
                self.test_results['puf_tests'][puf_type] = test_results
                print(f"  ✓ {puf_type.upper()} PUF test passed")
                
            except Exception as e:
                error_msg = f"Error testing {puf_type} PUF: {str(e)}"
                print(f"  ✗ {error_msg}")
                self.test_results['error_log'].append(error_msg)
                self.test_results['puf_tests'][puf_type] = {
                    'creation_successful': False,
                    'error': str(e)
                }
    
    def test_visualization_functions(self):
        """Test all visualization functions in PUFAnalyzer."""
        print("\n=== Testing Visualization Functions ===")
        
        puf_types = ['arbiter', 'sram', 'ring_oscillator']
        
        for puf_type in puf_types:
            print(f"\nTesting visualizations for {puf_type.upper()} PUF...")
            
            try:
                # Create test PUFs
                pufs = self.create_test_pufs(puf_type, num_pufs=8)
                analyzer = PUFAnalyzer(pufs[0])
                
                puf_viz_dir = f"{self.output_dir}/visualization_tests/{puf_type}"
                os.makedirs(puf_viz_dir, exist_ok=True)
                
                viz_results = {}
                
                # Test 1: Uniqueness Analysis (Matplotlib & Plotly)
                print(f"  Testing uniqueness analysis...")
                try:
                    start_time = time.time()
                    uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=500)
                    
                    # Test Matplotlib visualization
                    analyzer.plot_uniqueness_analysis(
                        uniqueness_data, 
                        save_path=f"{puf_viz_dir}/uniqueness_matplotlib.png",
                        use_plotly=False
                    )
                    
                    # Test Plotly visualization
                    analyzer.plot_uniqueness_analysis(
                        uniqueness_data, 
                        save_path=f"{puf_viz_dir}/uniqueness_plotly.html",
                        use_plotly=True
                    )
                    
                    viz_results['uniqueness_analysis'] = {
                        'success': True,
                        'matplotlib_generated': os.path.exists(f"{puf_viz_dir}/uniqueness_matplotlib.png"),
                        'plotly_generated': os.path.exists(f"{puf_viz_dir}/uniqueness_plotly.html"),
                        'mean_uniqueness': float(uniqueness_data['mean_uniqueness']),
                        'std_uniqueness': float(uniqueness_data['std_uniqueness']),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ Uniqueness analysis completed")
                    
                except Exception as e:
                    viz_results['uniqueness_analysis'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ Uniqueness analysis failed: {e}")
                
                # Test 2: Bit-Aliasing Analysis (Matplotlib & Plotly)
                print(f"  Testing bit-aliasing analysis...")
                try:
                    start_time = time.time()
                    aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=500)
                    
                    # Test Matplotlib visualization
                    analyzer.plot_bit_aliasing_analysis(
                        aliasing_data, 
                        save_path=f"{puf_viz_dir}/bit_aliasing_matplotlib.png",
                        use_plotly=False
                    )
                    
                    # Test Plotly visualization
                    analyzer.plot_bit_aliasing_analysis(
                        aliasing_data, 
                        save_path=f"{puf_viz_dir}/bit_aliasing_plotly.html",
                        use_plotly=True
                    )
                    
                    viz_results['bit_aliasing_analysis'] = {
                        'success': True,
                        'matplotlib_generated': os.path.exists(f"{puf_viz_dir}/bit_aliasing_matplotlib.png"),
                        'plotly_generated': os.path.exists(f"{puf_viz_dir}/bit_aliasing_plotly.html"),
                        'mean_aliasing': float(aliasing_data['mean_aliasing']),
                        'max_aliasing': float(aliasing_data['max_aliasing']),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ Bit-aliasing analysis completed")
                    
                except Exception as e:
                    viz_results['bit_aliasing_analysis'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ Bit-aliasing analysis failed: {e}")
                
                # Test 3: Reliability Analysis
                print(f"  Testing reliability analysis...")
                try:
                    start_time = time.time()
                    challenge = np.random.randint(0, 2, size=getattr(pufs[0], 'n_stages', 64))
                    reliability_data = analyzer.analyze_reliability_under_stress(
                        challenge=challenge,
                        num_trials=20,
                        time_points=np.linspace(0, 500, 10)
                    )
                    
                    analyzer.plot_reliability_analysis(
                        reliability_data, 
                        save_path=f"{puf_viz_dir}/reliability_analysis.png"
                    )
                    
                    viz_results['reliability_analysis'] = {
                        'success': True,
                        'file_generated': os.path.exists(f"{puf_viz_dir}/reliability_analysis.png"),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ Reliability analysis completed")
                    
                except Exception as e:
                    viz_results['reliability_analysis'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ Reliability analysis failed: {e}")
                
                # Test 4: Environmental Sensitivity Analysis
                print(f"  Testing environmental sensitivity analysis...")
                try:
                    start_time = time.time()
                    challenge = np.random.randint(0, 2, size=getattr(pufs[0], 'n_stages', 64))
                    sensitivity_data = analyzer.analyze_environmental_sensitivity(
                        challenge=challenge,
                        environment=MilitaryEnvironment.GROUND_MOBILE,
                        num_samples=100
                    )
                    
                    analyzer.plot_environmental_sensitivity(
                        sensitivity_data, 
                        save_path=f"{puf_viz_dir}/environmental_sensitivity.png"
                    )
                    
                    viz_results['environmental_sensitivity'] = {
                        'success': True,
                        'file_generated': os.path.exists(f"{puf_viz_dir}/environmental_sensitivity.png"),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ Environmental sensitivity analysis completed")
                    
                except Exception as e:
                    viz_results['environmental_sensitivity'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ Environmental sensitivity analysis failed: {e}")
                
                # Test 5: 3D PUF Response Analysis
                print(f"  Testing 3D PUF response analysis...")
                try:
                    start_time = time.time()
                    analyzer.plot_3d_puf_response_analysis(
                        pufs=pufs[:5],  # Use fewer PUFs for performance
                        num_crps=200,
                        save_path=f"{puf_viz_dir}/3d_puf_response.html"
                    )
                    
                    viz_results['3d_puf_response'] = {
                        'success': True,
                        'file_generated': os.path.exists(f"{puf_viz_dir}/3d_puf_response.html"),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ 3D PUF response analysis completed")
                    
                except Exception as e:
                    viz_results['3d_puf_response'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ 3D PUF response analysis failed: {e}")
                
                # Test 6: 3D Environmental Stress Impact
                print(f"  Testing 3D environmental stress impact...")
                try:
                    start_time = time.time()
                    analyzer.plot_3d_environmental_stress_impact(
                        puf=pufs[0],
                        environment=MilitaryEnvironment.GROUND_MOBILE,
                        save_path=f"{puf_viz_dir}/3d_environmental_stress.html"
                    )
                    
                    viz_results['3d_environmental_stress'] = {
                        'success': True,
                        'file_generated': os.path.exists(f"{puf_viz_dir}/3d_environmental_stress.html"),
                        'generation_time': time.time() - start_time
                    }
                    print(f"    ✓ 3D environmental stress analysis completed")
                    
                except Exception as e:
                    viz_results['3d_environmental_stress'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ 3D environmental stress analysis failed: {e}")
                
                # Test 7: 3D Threat Landscape (if attacks available)
                print(f"  Testing 3D threat landscape...")
                try:
                    attacks = self.create_test_attacks()
                    if attacks:
                        start_time = time.time()
                        analyzer.plot_3d_threat_landscape(
                            pufs=pufs[:3],  # Use fewer PUFs for performance
                            attacks=attacks[:3],  # Use fewer attacks for performance
                            environment=MilitaryEnvironment.GROUND_MOBILE,
                            save_path=f"{puf_viz_dir}/3d_threat_landscape.html"
                        )
                        
                        viz_results['3d_threat_landscape'] = {
                            'success': True,
                            'file_generated': os.path.exists(f"{puf_viz_dir}/3d_threat_landscape.html"),
                            'generation_time': time.time() - start_time
                        }
                        print(f"    ✓ 3D threat landscape analysis completed")
                    else:
                        viz_results['3d_threat_landscape'] = {'success': False, 'error': 'No attacks available'}
                        print(f"    ⚠ 3D threat landscape skipped (no attacks)")
                    
                except Exception as e:
                    viz_results['3d_threat_landscape'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ 3D threat landscape analysis failed: {e}")
                
                # Test 8: 3D Multi-Attack Comparison (if attacks available)
                print(f"  Testing 3D multi-attack comparison...")
                try:
                    attacks = self.create_test_attacks()
                    if attacks:
                        start_time = time.time()
                        analyzer.plot_3d_multi_attack_comparison(
                            pufs=pufs[:3],  # Use fewer PUFs for performance
                            attacks=attacks[:3],  # Use fewer attacks for performance
                            save_path=f"{puf_viz_dir}/3d_multi_attack.html"
                        )
                        
                        viz_results['3d_multi_attack'] = {
                            'success': True,
                            'file_generated': os.path.exists(f"{puf_viz_dir}/3d_multi_attack.html"),
                            'generation_time': time.time() - start_time
                        }
                        print(f"    ✓ 3D multi-attack comparison completed")
                    else:
                        viz_results['3d_multi_attack'] = {'success': False, 'error': 'No attacks available'}
                        print(f"    ⚠ 3D multi-attack comparison skipped (no attacks)")
                    
                except Exception as e:
                    viz_results['3d_multi_attack'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ 3D multi-attack comparison failed: {e}")
                
                # Test 9: Comprehensive Report Generation
                print(f"  Testing comprehensive report generation...")
                try:
                    start_time = time.time()
                    comprehensive_results = analyzer.generate_comprehensive_report(
                        pufs=pufs,
                        environment=MilitaryEnvironment.GROUND_MOBILE,
                        num_crps=500,
                        save_dir=f"{puf_viz_dir}/comprehensive_report",
                        use_plotly=False
                    )
                    
                    viz_results['comprehensive_report'] = {
                        'success': True,
                        'metrics_generated': len(comprehensive_results) > 0,
                        'generation_time': time.time() - start_time,
                        'sample_metrics': {k: v for k, v in list(comprehensive_results.items())[:3]}
                    }
                    print(f"    ✓ Comprehensive report generation completed")
                    
                except Exception as e:
                    viz_results['comprehensive_report'] = {'success': False, 'error': str(e)}
                    print(f"    ✗ Comprehensive report generation failed: {e}")
                
                self.test_results['visualization_tests'][puf_type] = viz_results
                
            except Exception as e:
                error_msg = f"Error testing visualizations for {puf_type}: {str(e)}"
                print(f"  ✗ {error_msg}")
                self.test_results['error_log'].append(error_msg)
    
    def test_threat_assessment_reports(self):
        """Test threat assessment report generation."""
        print("\n=== Testing Threat Assessment Reports ===")
        
        try:
            # Create test PUFs
            pufs = self.create_test_pufs('arbiter', num_pufs=5)
            
            # Test different military environments
            environments = [
                MilitaryEnvironment.GROUND_MOBILE,
                MilitaryEnvironment.AIRCRAFT_INTERNAL,
                MilitaryEnvironment.SPACE_VEHICLE
            ]
            
            threat_results = {}
            
            for env in environments:
                print(f"Testing threat assessment for {env.value}...")
                
                try:
                    start_time = time.time()
                    
                    # Create threat assessment generator
                    report_gen = ThreatAssessmentReportGenerator(classification_level="UNCLASSIFIED")
                    
                    # Generate report
                    report_data = report_gen.generate_security_assessment_report(
                        pufs=pufs,
                        environment=env,
                        use_cases=[f"{env.value.replace('_', ' ').title()} Security"],
                        output_dir=f"{self.output_dir}/threat_assessment_{env.value}"
                    )
                    
                    threat_results[env.value] = {
                        'success': True,
                        'report_generated': True,
                        'generation_time': time.time() - start_time,
                        'sample_metrics': {k: v for k, v in list(report_data.items())[:3] if isinstance(v, (int, float, str))}
                    }
                    print(f"  ✓ Threat assessment for {env.value} completed")
                    
                except Exception as e:
                    threat_results[env.value] = {'success': False, 'error': str(e)}
                    print(f"  ✗ Threat assessment for {env.value} failed: {e}")
            
            self.test_results['threat_assessment'] = threat_results
            
        except Exception as e:
            error_msg = f"Error in threat assessment testing: {str(e)}"
            print(f"✗ {error_msg}")
            self.test_results['error_log'].append(error_msg)
    
    def generate_thesis_data_samples(self):
        """Generate sample datasets for thesis use."""
        print("\n=== Generating Thesis Data Samples ===")
        
        try:
            thesis_dir = f"{self.output_dir}/thesis_data"
            
            # Generate sample data for each PUF type
            puf_types = ['arbiter', 'sram', 'ring_oscillator']
            
            for puf_type in puf_types:
                print(f"Generating thesis data for {puf_type.upper()} PUF...")
                
                # Create PUFs
                pufs = self.create_test_pufs(puf_type, num_pufs=15)
                analyzer = PUFAnalyzer(pufs[0])
                
                puf_thesis_dir = f"{thesis_dir}/{puf_type}"
                os.makedirs(puf_thesis_dir, exist_ok=True)
                
                # Generate high-quality visualizations for thesis
                uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=1000)
                aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=1000)
                
                # Save high-resolution figures
                analyzer.plot_uniqueness_analysis(
                    uniqueness_data, 
                    save_path=f"{puf_thesis_dir}/thesis_uniqueness_analysis.png",
                    use_plotly=False
                )
                
                analyzer.plot_bit_aliasing_analysis(
                    aliasing_data, 
                    save_path=f"{puf_thesis_dir}/thesis_bit_aliasing_analysis.png",
                    use_plotly=False
                )
                
                # Generate interactive versions
                analyzer.plot_uniqueness_analysis(
                    uniqueness_data, 
                    save_path=f"{puf_thesis_dir}/thesis_uniqueness_interactive.html",
                    use_plotly=True
                )
                
                analyzer.plot_bit_aliasing_analysis(
                    aliasing_data, 
                    save_path=f"{puf_thesis_dir}/thesis_bit_aliasing_interactive.html",
                    use_plotly=True
                )
                
                # Save raw data for further analysis
                thesis_data = {
                    'uniqueness_metrics': {
                        'mean_uniqueness': float(uniqueness_data['mean_uniqueness']),
                        'std_uniqueness': float(uniqueness_data['std_uniqueness']),
                        'pairwise_distances': uniqueness_data['pairwise_distances'].tolist()
                    },
                    'aliasing_metrics': {
                        'mean_aliasing': float(aliasing_data['mean_aliasing']),
                        'max_aliasing': float(aliasing_data['max_aliasing']),
                        'bit_frequencies': aliasing_data['bit_frequencies'].tolist()
                    }
                }
                
                with open(f"{puf_thesis_dir}/thesis_data_metrics.json", 'w') as f:
                    json.dump(thesis_data, f, indent=2)
                
                print(f"  ✓ Thesis data for {puf_type.upper()} generated")
            
            print("✓ All thesis data samples generated")
            
        except Exception as e:
            error_msg = f"Error generating thesis data: {str(e)}"
            print(f"✗ {error_msg}")
            self.test_results['error_log'].append(error_msg)
    
    def run_performance_analysis(self):
        """Analyze performance of visualization generation."""
        print("\n=== Running Performance Analysis ===")
        
        try:
            # Collect timing data from previous tests
            performance_data = {}
            
            for puf_type, viz_tests in self.test_results.get('visualization_tests', {}).items():
                total_time = 0
                successful_tests = 0
                
                for test_name, test_data in viz_tests.items():
                    if test_data.get('success', False) and 'generation_time' in test_data:
                        total_time += test_data['generation_time']
                        successful_tests += 1
                
                if successful_tests > 0:
                    performance_data[puf_type] = {
                        'total_time': total_time,
                        'average_time': total_time / successful_tests,
                        'successful_tests': successful_tests,
                        'performance_rating': 'excellent' if total_time < 30 else 'good' if total_time < 60 else 'needs_improvement'
                    }
            
            self.test_results['performance_metrics'] = performance_data
            
            # Save performance report
            with open(f"{self.output_dir}/performance_report.json", 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            print("✓ Performance analysis completed")
            
        except Exception as e:
            error_msg = f"Error in performance analysis: {str(e)}"
            print(f"✗ {error_msg}")
            self.test_results['error_log'].append(error_msg)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n=== Generating Summary Report ===")
        
        # Calculate summary statistics
        total_tests = 0
        successful_tests = 0
        
        # Count PUF tests
        for puf_type, test_data in self.test_results.get('puf_tests', {}).items():
            total_tests += 1
            if test_data.get('creation_successful', False):
                successful_tests += 1
        
        # Count visualization tests
        for puf_type, viz_tests in self.test_results.get('visualization_tests', {}).items():
            for test_name, test_data in viz_tests.items():
                total_tests += 1
                if test_data.get('success', False):
                    successful_tests += 1
        
        # Count threat assessment tests
        for env, test_data in self.test_results.get('threat_assessment', {}).items():
            total_tests += 1
            if test_data.get('success', False):
                successful_tests += 1
        
        summary = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_errors': len(self.test_results.get('error_log', [])),
            'test_categories': {
                'puf_types_tested': len(self.test_results.get('puf_tests', {})),
                'visualization_functions_tested': sum(len(viz_tests) for viz_tests in self.test_results.get('visualization_tests', {}).values()),
                'threat_assessments_tested': len(self.test_results.get('threat_assessment', {}))
            }
        }
        
        self.test_results['summary'] = summary
        
        # Save complete results
        with open(f"{self.output_dir}/comprehensive_test_results.json", 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate human-readable summary
        with open(f"{self.output_dir}/test_summary.txt", 'w') as f:
            f.write("PPET Comprehensive Visualization Test Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {self.test_results['timestamp']}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful Tests: {successful_tests}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"Total Errors: {summary['total_errors']}\n\n")
            
            f.write("Test Categories:\n")
            f.write(f"  PUF Types Tested: {summary['test_categories']['puf_types_tested']}\n")
            f.write(f"  Visualization Functions Tested: {summary['test_categories']['visualization_functions_tested']}\n")
            f.write(f"  Threat Assessments Tested: {summary['test_categories']['threat_assessments_tested']}\n\n")
            
            if self.test_results.get('error_log'):
                f.write("Error Log:\n")
                for error in self.test_results['error_log']:
                    f.write(f"  - {error}\n")
        
        print("✓ Summary report generated")
        return summary
    
    def run_comprehensive_test(self):
        """Run all comprehensive tests."""
        print("Starting comprehensive visualization testing...")
        
        try:
            # Test all components
            self.test_puf_types()
            self.test_visualization_functions()
            self.test_threat_assessment_reports()
            self.generate_thesis_data_samples()
            self.run_performance_analysis()
            summary = self.generate_summary_report()
            
            print("\n" + "=" * 60)
            print("COMPREHENSIVE TEST SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Successful Tests: {summary['successful_tests']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Errors: {summary['total_errors']}")
            
            if summary['success_rate'] >= 80:
                print("🎉 Overall Status: EXCELLENT")
            elif summary['success_rate'] >= 60:
                print("✅ Overall Status: GOOD")
            else:
                print("⚠️  Overall Status: NEEDS IMPROVEMENT")
            
            print(f"\nAll results saved to: {self.output_dir}/")
            print("Check comprehensive_test_results.json for detailed results")
            
        except Exception as e:
            print(f"Critical error in comprehensive test: {e}")
            traceback.print_exc()


def main():
    """Main function to run comprehensive visualization tests."""
    print("PPET Comprehensive Visualization Test Suite")
    print("=" * 60)
    
    # Create tester instance
    tester = ComprehensiveVisualizationTester()
    
    # Run all tests
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()