"""Test suite for military-grade PUF features."""

import pytest
import numpy as np
from ppet.core.puf_emulator import PUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF
from ppet.core.military_stressors import MilitaryStressors, MilitaryEnvironment
from ppet.core.threat_simulator import (
    MLAttack, EnhancedSideChannelAttack,
    SupplyChainAttack, FaultInjectionAttack
)
from ppet.core.analysis import PUFAnalyzer

@pytest.fixture
def military_arbiter_puf():
    """Create ArbiterPUF with military environment."""
    return ArbiterPUF(
        n_stages=64,
        seed=42,
        military_environment=MilitaryEnvironment.GROUND_MOBILE
    )

@pytest.fixture
def military_stressor():
    """Create military stressor simulator."""
    return MilitaryStressors(
        environment=MilitaryEnvironment.GROUND_MOBILE,
        mission_duration=1000.0,
        seed=42
    )

def test_military_stressor_initialization():
    """Test military stressor initialization."""
    stressor = MilitaryStressors(MilitaryEnvironment.GROUND_MOBILE)
    
    # Check profile initialization
    assert stressor.environment == MilitaryEnvironment.GROUND_MOBILE
    assert stressor.mission_duration == 1000.0
    
    # Check temperature profile
    profile = stressor.temp_profiles[MilitaryEnvironment.GROUND_MOBILE]
    assert profile.min_temp == -40.0
    assert profile.max_temp == 85.0

def test_temperature_cycling(military_stressor):
    """Test temperature cycling behavior."""
    # Test temperature at different mission times
    temps = [
        military_stressor.get_temperature_stress(t)
        for t in [0, 6, 12, 18, 24]
    ]
    
    # Check temperature range
    assert all(-40.0 <= t <= 85.0 for t in temps)
    
    # Check cycling behavior
    temp_0h = military_stressor.get_temperature_stress(0)
    temp_24h = military_stressor.get_temperature_stress(24)
    assert abs(temp_0h - temp_24h) < 1.0  # Should complete cycle

def test_emi_simulation(military_stressor):
    """Test EMI simulation."""
    emi_0h = military_stressor.get_emi_stress(0)
    
    # Check EMI parameters
    assert 'conducted' in emi_0h
    assert 'radiated' in emi_0h
    assert 'frequency' in emi_0h
    assert 'normalized' in emi_0h
    
    # Check value ranges
    assert 0 <= emi_0h['normalized'] <= 2.0  # Normalized EMI
    assert emi_0h['frequency'] >= 10e3  # Minimum frequency

def test_aging_effects(military_stressor):
    """Test aging effects calculation."""
    # Check aging progression
    aging_0h = military_stressor.get_aging_factor(0)
    aging_500h = military_stressor.get_aging_factor(500)
    aging_1000h = military_stressor.get_aging_factor(1000)
    
    assert aging_0h <= aging_500h <= aging_1000h
    assert aging_0h >= 1.0  # No negative aging

def test_puf_military_environment(military_arbiter_puf):
    """Test PUF behavior in military environment."""
    # Generate CRPs at different mission times
    challenge = np.random.randint(0, 2, size=64)
    
    responses = []
    for time in [0, 250, 500, 750, 1000]:
        military_arbiter_puf.update_mission_time(time)
        responses.append(military_arbiter_puf.evaluate(challenge))
    
    # Check response variation
    unique_responses = len(set(responses))
    assert unique_responses > 1  # Should see some variation

def test_ml_attack_with_environment():
    """Test ML attack with environmental consideration."""
    # Create PUF and generate training data
    puf = ArbiterPUF(
        n_stages=32,
        military_environment=MilitaryEnvironment.GROUND_MOBILE
    )
    
    challenges = np.random.randint(0, 2, size=(1000, 32))
    responses = np.array([puf.evaluate(c) for c in challenges])
    
    # Train attack with and without environmental augmentation
    attack_std = MLAttack(model_type='rf')
    attack_env = MLAttack(
        model_type='rf',
        environmental_augmentation=True,
        military_environment=MilitaryEnvironment.GROUND_MOBILE
    )
    
    attack_std.train(challenges, responses)
    attack_env.train(challenges, responses)
    
    # Generate test data at different environmental conditions
    test_challenges = np.random.randint(0, 2, size=(100, 32))
    puf.update_mission_time(500)  # Mid-mission conditions
    test_responses = np.array([puf.evaluate(c) for c in test_challenges])
    
    # Compare prediction accuracy
    acc_std = attack_std.evaluate(test_challenges, test_responses)
    acc_env = attack_env.evaluate(test_challenges, test_responses)
    
    assert acc_env >= acc_std  # Environmental model should be more accurate

def test_enhanced_side_channel_attack():
    """Test enhanced side-channel attack with military considerations."""
    # Create attack instances
    attack_std = EnhancedSideChannelAttack(
        attack_type='power',
        noise_std=0.1
    )
    attack_mil = EnhancedSideChannelAttack(
        attack_type='power',
        noise_std=0.1,
        military_environment=MilitaryEnvironment.GROUND_MOBILE
    )
    
    # Generate test data
    challenges = np.random.randint(0, 2, size=(100, 32))
    responses = np.random.randint(0, 2, size=100)
    
    # Train both attacks
    attack_std.train(challenges, responses)
    attack_mil.train(challenges, responses)
    
    # Compare measurements
    meas_std = attack_std._simulate_measurements(challenges[:1])
    meas_mil = attack_mil._simulate_measurements(challenges[:1], time=500)
    
    assert not np.allclose(meas_std, meas_mil)  # Should differ due to environment

def test_supply_chain_attack():
    """Test supply chain attack simulation."""
    attack = SupplyChainAttack(tampering_rate=0.05)
    
    # Generate test data
    challenges = np.random.randint(0, 2, size=(100, 64))
    responses = np.random.randint(0, 2, size=100)
    
    # Train attack
    attack.train(challenges, responses)
    
    # Verify tampering
    assert len(attack.tampered_indices) == int(0.05 * 64)
    
    # Test predictions
    pred = attack.predict(challenges)
    assert len(pred) == len(challenges)

def test_fault_injection_attack():
    """Test fault injection attack simulation."""
    attack = FaultInjectionAttack(
        injection_type='voltage',
        precision=0.9,
        strength=0.8
    )
    
    # Generate test data
    challenges = np.random.randint(0, 2, size=(100, 32))
    responses = np.random.randint(0, 2, size=100)
    
    # Train attack
    attack.train(challenges, responses)
    
    # Verify vulnerable stages identified
    assert attack.vulnerable_stages is not None
    assert len(attack.vulnerable_stages) > 0
    
    # Test predictions
    pred = attack.predict(challenges)
    assert len(pred) == len(challenges)

def test_analyzer_military_metrics():
    """Test PUF analyzer with military-grade metrics."""
    # Create PUF and analyzer
    puf = ArbiterPUF(
        n_stages=32,
        military_environment=MilitaryEnvironment.GROUND_MOBILE
    )
    analyzer = PUFAnalyzer(puf)
    
    # Generate test challenge
    challenge = np.random.randint(0, 2, size=32)
    
    # Analyze reliability
    analysis = analyzer.analyze_reliability_under_stress(
        challenge,
        num_trials=50,
        time_points=[0, 250, 500, 750, 1000]
    )
    
    # Check analysis results
    assert 'temperatures' in analysis
    assert 'emi_levels' in analysis
    assert 'aging_factors' in analysis
    assert len(analysis['temperatures']) == 5
    
    # Generate report
    report = analyzer.generate_reliability_report(
        challenge,
        MilitaryEnvironment.GROUND_MOBILE
    )
    
    # Check report metrics
    assert 'mean_reliability_percent' in report
    assert 'worst_case_reliability_percent' in report
    assert 'temperature_sensitivity_percent' in report
    assert 'emi_sensitivity_percent' in report
    assert 'aging_impact' in report 