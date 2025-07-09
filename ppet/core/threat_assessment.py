"""
Comprehensive threat assessment report generation for defense applications.

This module provides automated threat assessment reporting capabilities
for military and defense PUF deployments.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from .puf_emulator import PUF
from .analysis import PUFAnalyzer
from .military_stressors import MilitaryEnvironment, MilitaryStressors
from .threat_simulator import Attack, MLAttack, SideChannelAttack, SupplyChainAttack, FaultInjectionAttack

class ThreatAssessmentReportGenerator:
    """Generate comprehensive threat assessment reports for defense applications."""
    
    def __init__(self, classification_level: str = "UNCLASSIFIED"):
        """Initialize threat assessment report generator.
        
        Args:
            classification_level: Security classification level
        """
        self.classification_level = classification_level
        self.report_timestamp = datetime.now()
        
    def generate_defense_procurement_report(
        self,
        pufs: List[PUF],
        environment: MilitaryEnvironment,
        use_cases: List[str],
        output_dir: str = "threat_assessment_report"
    ) -> Dict:
        """Generate comprehensive defense procurement report.
        
        Args:
            pufs: List of PUF instances to evaluate
            environment: Military environment profile
            use_cases: List of use case names
            output_dir: Directory to save report files
            
        Returns:
            Dictionary containing all assessment metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize report data
        report_data = {
            'classification': self.classification_level,
            'timestamp': self.report_timestamp.isoformat(),
            'environment': environment.value,
            'use_cases': use_cases,
            'assessment_summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # Create analyzer
        analyzer = PUFAnalyzer(pufs[0])
        
        # 1. Uniqueness Assessment
        print("Conducting uniqueness assessment...")
        uniqueness_data = analyzer.analyze_uniqueness(pufs, num_crps=1000)
        report_data['assessment_summary']['uniqueness_score'] = uniqueness_data['mean_uniqueness']
        report_data['assessment_summary']['uniqueness_std'] = uniqueness_data['std_uniqueness']
        
        # Generate uniqueness visualization
        analyzer.plot_uniqueness_analysis(
            uniqueness_data,
            save_path=f"{output_dir}/uniqueness_analysis.html",
            use_plotly=True
        )
        
        # 2. Reliability Assessment
        print("Conducting reliability assessment...")
        test_challenge = np.random.randint(0, 2, size=getattr(pufs[0], 'n_stages', 64))
        reliability_metrics = analyzer.generate_reliability_report(
            challenge=test_challenge,
            environment=environment,
            save_path=f"{output_dir}/reliability"
        )
        report_data['assessment_summary'].update(reliability_metrics)
        
        # 3. Bit-Aliasing Assessment
        print("Conducting bit-aliasing assessment...")
        aliasing_data = analyzer.analyze_bit_aliasing(pufs, num_crps=1000)
        report_data['assessment_summary']['bit_aliasing_score'] = aliasing_data['mean_aliasing']
        report_data['assessment_summary']['max_bit_aliasing'] = aliasing_data['max_aliasing']
        
        # Generate bit-aliasing visualization
        analyzer.plot_bit_aliasing_analysis(
            aliasing_data,
            save_path=f"{output_dir}/bit_aliasing_analysis.html",
            use_plotly=True
        )
        
        # 4. Threat Landscape Assessment
        print("Conducting threat landscape assessment...")
        attacks = self._create_defense_attack_suite(environment)
        
        # Generate 3D threat landscape
        analyzer.plot_3d_threat_landscape(
            pufs=pufs[:5],  # Use first 5 PUFs for efficiency
            attacks=attacks,
            environment=environment,
            save_path=f"{output_dir}/threat_landscape_3d.html"
        )
        
        # 5. Multi-Attack Comparison
        print("Conducting multi-attack comparison...")
        analyzer.plot_3d_multi_attack_comparison(
            pufs=pufs[:3],
            attacks=attacks,
            save_path=f"{output_dir}/multi_attack_comparison_3d.html"
        )
        
        # 6. Environmental Stress Assessment
        print("Conducting environmental stress assessment...")
        analyzer.plot_3d_environmental_stress_impact(
            puf=pufs[0],
            environment=environment,
            save_path=f"{output_dir}/environmental_stress_3d.html"
        )
        
        # 7. Attack Success Rate Analysis
        print("Analyzing attack success rates...")
        attack_results = self._evaluate_attack_effectiveness(pufs, attacks)
        report_data['detailed_analysis']['attack_success_rates'] = attack_results
        
        # 8. Generate Risk Assessment
        print("Generating risk assessment...")
        risk_assessment = self._generate_risk_assessment(report_data['assessment_summary'], attack_results)
        report_data['assessment_summary']['overall_risk_score'] = risk_assessment['overall_risk']
        report_data['assessment_summary']['risk_level'] = risk_assessment['risk_level']
        
        # 9. Generate Recommendations
        print("Generating recommendations...")
        recommendations = self._generate_recommendations(report_data['assessment_summary'], environment)
        report_data['recommendations'] = recommendations
        
        # 10. Generate Executive Summary
        print("Generating executive summary...")
        executive_summary = self._generate_executive_summary(report_data)
        report_data['executive_summary'] = executive_summary
        
        # Save detailed report
        with open(f"{output_dir}/threat_assessment_report.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report_data, f"{output_dir}/threat_assessment_report.html")
        
        print(f"Threat assessment report generated in: {output_dir}/")
        
        return report_data
    
    def _create_defense_attack_suite(self, environment: MilitaryEnvironment) -> List[Attack]:
        """Create comprehensive attack suite for defense evaluation."""
        attacks = []
        
        # Machine Learning Attacks
        attacks.append(MLAttack(
            model_type='rf',
            environmental_augmentation=True,
            military_environment=environment
        ))
        attacks.append(MLAttack(model_type='mlp', environmental_augmentation=False))
        
        # Side-Channel Attacks
        attacks.append(SideChannelAttack(attack_type='power', noise_std=0.1))
        attacks.append(SideChannelAttack(attack_type='timing', noise_std=0.05))
        
        # Supply Chain Attacks
        attacks.append(SupplyChainAttack(tampering_rate=0.01, detection_difficulty=0.9))
        attacks.append(SupplyChainAttack(tampering_rate=0.05, detection_difficulty=0.7))
        
        # Fault Injection Attacks
        attacks.append(FaultInjectionAttack(injection_type='voltage', precision=0.8))
        attacks.append(FaultInjectionAttack(injection_type='laser', precision=0.9))
        
        return attacks
    
    def _evaluate_attack_effectiveness(self, pufs: List[PUF], attacks: List[Attack]) -> Dict:
        """Evaluate effectiveness of each attack."""
        results = {}
        
        # Generate test data
        test_challenges = np.random.randint(0, 2, size=(500, getattr(pufs[0], 'n_stages', 64)))
        
        for attack in attacks:
            try:
                # Get ground truth responses
                if hasattr(pufs[0], 'generate_responses'):
                    true_responses = pufs[0].generate_responses(test_challenges)
                else:
                    true_responses = pufs[0].generate_crps(len(test_challenges))
                    if isinstance(true_responses, tuple):
                        true_responses = true_responses[1]
                
                # Train and evaluate attack
                attack.train(test_challenges, true_responses)
                success_rate = attack.evaluate(test_challenges, true_responses)
                
                results[attack.name] = {
                    'success_rate': success_rate,
                    'threat_level': self._classify_threat_level(success_rate),
                    'countermeasures': self._suggest_countermeasures(attack.name, success_rate)
                }
                
            except Exception as e:
                results[attack.name] = {
                    'success_rate': 0.0,
                    'threat_level': 'UNKNOWN',
                    'error': str(e)
                }
        
        return results
    
    def _classify_threat_level(self, success_rate: float) -> str:
        """Classify threat level based on success rate."""
        if success_rate >= 0.8:
            return 'CRITICAL'
        elif success_rate >= 0.6:
            return 'HIGH'
        elif success_rate >= 0.4:
            return 'MEDIUM'
        elif success_rate >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _suggest_countermeasures(self, attack_name: str, success_rate: float) -> List[str]:
        """Suggest countermeasures based on attack type and success rate."""
        countermeasures = []
        
        if 'ML' in attack_name:
            countermeasures.append("Implement challenge obfuscation")
            countermeasures.append("Use environmental variation during operation")
            if success_rate > 0.6:
                countermeasures.append("Consider hybrid PUF architectures")
        
        elif 'SCA' in attack_name:
            countermeasures.append("Implement power analysis countermeasures")
            countermeasures.append("Add noise injection during evaluation")
            if success_rate > 0.7:
                countermeasures.append("Use differential power analysis protection")
        
        elif 'SUPPLY_CHAIN' in attack_name:
            countermeasures.append("Implement hardware authentication")
            countermeasures.append("Use trusted foundry sources")
            countermeasures.append("Add tamper detection mechanisms")
        
        elif 'FAULT' in attack_name:
            countermeasures.append("Implement fault detection circuits")
            countermeasures.append("Use redundant evaluation")
            countermeasures.append("Add temporal and spatial diversity")
        
        return countermeasures
    
    def _generate_risk_assessment(self, summary: Dict, attack_results: Dict) -> Dict:
        """Generate overall risk assessment."""
        risk_factors = []
        
        # Uniqueness risk
        if summary['uniqueness_score'] < 0.45:
            risk_factors.append(('LOW_UNIQUENESS', 0.3))
        elif summary['uniqueness_score'] < 0.48:
            risk_factors.append(('MODERATE_UNIQUENESS', 0.1))
        
        # Reliability risk
        if summary['mean_reliability_percent'] < 90:
            risk_factors.append(('LOW_RELIABILITY', 0.4))
        elif summary['mean_reliability_percent'] < 95:
            risk_factors.append(('MODERATE_RELIABILITY', 0.2))
        
        # Attack success risk
        max_attack_success = max([r['success_rate'] for r in attack_results.values()])
        if max_attack_success > 0.7:
            risk_factors.append(('HIGH_ATTACK_SUCCESS', 0.5))
        elif max_attack_success > 0.5:
            risk_factors.append(('MODERATE_ATTACK_SUCCESS', 0.3))
        
        # Calculate overall risk
        overall_risk = sum([weight for _, weight in risk_factors])
        
        if overall_risk > 0.8:
            risk_level = 'CRITICAL'
        elif overall_risk > 0.6:
            risk_level = 'HIGH'
        elif overall_risk > 0.4:
            risk_level = 'MEDIUM'
        elif overall_risk > 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _generate_recommendations(self, summary: Dict, environment: MilitaryEnvironment) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        # Uniqueness recommendations
        if summary['uniqueness_score'] < 0.45:
            recommendations.append("CRITICAL: Improve manufacturing process variation")
            recommendations.append("Consider alternative PUF architectures")
        
        # Reliability recommendations
        if summary['mean_reliability_percent'] < 90:
            recommendations.append("CRITICAL: Implement error correction mechanisms")
            recommendations.append("Reduce environmental sensitivity")
        
        # Environment-specific recommendations
        if environment == MilitaryEnvironment.SPACE_VEHICLE:
            recommendations.append("Implement radiation-hardened design")
            recommendations.append("Use triple modular redundancy")
        elif environment == MilitaryEnvironment.GROUND_MOBILE:
            recommendations.append("Implement temperature compensation")
            recommendations.append("Add vibration resistance")
        
        # Attack-specific recommendations
        if summary['overall_risk_score'] > 0.6:
            recommendations.append("CRITICAL: Implement multi-layer security")
            recommendations.append("Use dynamic challenge generation")
        
        return recommendations
    
    def _generate_executive_summary(self, report_data: Dict) -> str:
        """Generate executive summary."""
        summary = f"""
EXECUTIVE SUMMARY - {self.classification_level}

Assessment Date: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Environment: {report_data['environment'].upper()}

OVERALL ASSESSMENT:
Risk Level: {report_data['assessment_summary']['risk_level']}
Overall Risk Score: {report_data['assessment_summary']['overall_risk_score']:.2f}

KEY METRICS:
• Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}
• Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%
• Bit-Aliasing Score: {report_data['assessment_summary']['bit_aliasing_score']:.3f}

THREAT LANDSCAPE:
• {len(report_data['detailed_analysis']['attack_success_rates'])} attack vectors evaluated
• Highest threat: {self._get_highest_threat(report_data['detailed_analysis']['attack_success_rates'])}

RECOMMENDATIONS:
{chr(10).join(['• ' + rec for rec in report_data['recommendations'][:5]])}

DEPLOYMENT READINESS:
{self._get_deployment_readiness(report_data['assessment_summary']['risk_level'])}
"""
        return summary
    
    def _get_highest_threat(self, attack_results: Dict) -> str:
        """Identify highest threat from attack results."""
        max_threat = max(attack_results.items(), key=lambda x: x[1]['success_rate'])
        return f"{max_threat[0]} ({max_threat[1]['success_rate']:.1%} success rate)"
    
    def _get_deployment_readiness(self, risk_level: str) -> str:
        """Determine deployment readiness."""
        if risk_level == 'MINIMAL':
            return "READY FOR DEPLOYMENT"
        elif risk_level == 'LOW':
            return "READY FOR DEPLOYMENT WITH MONITORING"
        elif risk_level == 'MEDIUM':
            return "CONDITIONAL DEPLOYMENT - IMPLEMENT COUNTERMEASURES"
        elif risk_level == 'HIGH':
            return "NOT READY - MAJOR SECURITY CONCERNS"
        else:
            return "NOT READY - CRITICAL SECURITY VULNERABILITIES"
    
    def _generate_html_report(self, report_data: Dict, output_path: str):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PUF Threat Assessment Report - {self.classification_level}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .risk-critical {{ color: #dc3545; font-weight: bold; }}
        .risk-high {{ color: #fd7e14; font-weight: bold; }}
        .risk-medium {{ color: #ffc107; font-weight: bold; }}
        .risk-low {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PUF Threat Assessment Report</h1>
        <p><strong>Classification:</strong> {report_data['classification']}</p>
        <p><strong>Date:</strong> {report_data['timestamp']}</p>
        <p><strong>Environment:</strong> {report_data['environment'].upper()}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <pre>{report_data['executive_summary']}</pre>
    </div>
    
    <div class="section">
        <h2>Assessment Summary</h2>
        <div class="metric">Overall Risk Level: <span class="risk-{report_data['assessment_summary']['risk_level'].lower()}">{report_data['assessment_summary']['risk_level']}</span></div>
        <div class="metric">Uniqueness Score: {report_data['assessment_summary']['uniqueness_score']:.3f}</div>
        <div class="metric">Mean Reliability: {report_data['assessment_summary']['mean_reliability_percent']:.1f}%</div>
        <div class="metric">Bit-Aliasing Score: {report_data['assessment_summary']['bit_aliasing_score']:.3f}</div>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {''.join([f'<div class="recommendation">{rec}</div>' for rec in report_data['recommendations']])}
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <ul>
            <li><a href="uniqueness_analysis.html">Uniqueness Analysis (3D)</a></li>
            <li><a href="bit_aliasing_analysis.html">Bit-Aliasing Analysis (3D)</a></li>
            <li><a href="threat_landscape_3d.html">Threat Landscape (3D)</a></li>
            <li><a href="multi_attack_comparison_3d.html">Multi-Attack Comparison (3D)</a></li>
            <li><a href="environmental_stress_3d.html">Environmental Stress Impact (3D)</a></li>
        </ul>
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)