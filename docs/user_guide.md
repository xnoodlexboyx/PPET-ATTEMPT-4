# PPET User Guide

This guide provides detailed instructions for using the Defense-Oriented Physical Unclonable Function Emulation and Analysis Tool (PPET).

## Table of Contents

1. [Introduction](#introduction)
2. [PUF Emulation](#puf-emulation)
3. [Military Environment Simulation](#military-environment-simulation)
4. [Security Analysis](#security-analysis)
5. [Use Cases](#use-cases)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)
8. [Military Standards Compliance](#military-standards-compliance)

## Introduction

PPET is a defense-oriented framework designed to help researchers and developers work with Physical Unclonable Functions (PUFs) in military and national security applications by providing:
- Accurate PUF emulation with military-grade environmental modeling
- Comprehensive security analysis for defense applications
- Military-specific use case implementations
- Extensible framework for custom defense applications

## PUF Emulation

### Arbiter PUF with Military Environment

The Arbiter PUF emulator now supports military-grade environmental conditions:

```python
from ppet.core.puf_emulator import ArbiterPUF
from ppet.core.military_stressors import MilitaryEnvironment

# Create PUF with military environment
puf = ArbiterPUF(
    n_stages=64,
    military_environment=MilitaryEnvironment.GROUND_MOBILE,
    mission_time=0.0  # Initial mission time
)

# Generate CRPs with environmental updates
puf.update_mission_time(500)  # Update to 500 hours
challenges, responses = puf.generate_crps(num_crps=1000)
```

### SRAM PUF

The SRAM PUF emulator with enhanced environmental modeling:

```python
from ppet.core.puf_emulator import SRAMPUF

# Create SRAM PUF with military environment
puf = SRAMPUF(
    array_size=(128, 128),
    military_environment=MilitaryEnvironment.AIRCRAFT_INTERNAL
)

# Generate startup values under stress
responses = puf.generate_responses()
```

### Ring Oscillator PUF

The Ring Oscillator PUF with military-grade features:

```python
from ppet.core.puf_emulator import RingOscillatorPUF

# Create RO-PUF with military environment
puf = RingOscillatorPUF(
    num_oscillators=256,
    military_environment=MilitaryEnvironment.NAVAL_SHELTERED
)

# Generate responses under environmental stress
responses = puf.generate_responses(num_pairs=100)
```

## Military Environment Simulation

PPET provides comprehensive military environment simulation:

```python
from ppet.core.military_stressors import MilitaryStressors, MilitaryEnvironment

# Create military stressor simulator
stressor = MilitaryStressors(
    environment=MilitaryEnvironment.GROUND_MOBILE,
    mission_duration=1000.0
)

# Get environmental conditions
conditions = stressor.get_all_stressors(time=500)  # At 500 hours
print(f"Temperature: {conditions['temperature']:.1f}°C")
print(f"EMI Level: {conditions['em_noise']:.3f}")
print(f"Aging Factor: {conditions['aging_factor']:.3f}")

# Available military environments
environments = [
    MilitaryEnvironment.GROUND_MOBILE,     # -40°C to +85°C
    MilitaryEnvironment.AIRCRAFT_INTERNAL, # -45°C to +70°C
    MilitaryEnvironment.AIRCRAFT_EXTERNAL, # -55°C to +125°C
    MilitaryEnvironment.NAVAL_SHELTERED,   # -10°C to +65°C
    MilitaryEnvironment.NAVAL_EXPOSED,     # -25°C to +55°C
    MilitaryEnvironment.SPACE_VEHICLE      # -65°C to +125°C
]
```

## Security Analysis

### Enhanced PUF Quality Metrics

Analyze PUF characteristics under military conditions:

```python
from ppet.core.analysis import PUFAnalyzer

# Create analyzer with military PUF
analyzer = PUFAnalyzer(puf)

# Analyze reliability under stress
analysis = analyzer.analyze_reliability_under_stress(
    challenge,
    num_trials=100,
    time_points=[0, 250, 500, 750, 1000]
)

# Generate military-grade report
report = analyzer.generate_reliability_report(
    challenge,
    MilitaryEnvironment.GROUND_MOBILE
)

# Visualize military metrics
analyzer.plot_reliability_analysis(analysis, "reliability.png")
analyzer.plot_environmental_sensitivity(analysis, "sensitivity.png")
```

### Advanced Threat Simulation

Evaluate PUF security against military-grade attacks:

```python
from ppet.core.threat_simulator import (
    MLAttack, EnhancedSideChannelAttack,
    SupplyChainAttack, FaultInjectionAttack
)

# Enhanced ML Attack with environmental awareness
ml_attack = MLAttack(
    model_type='rf',
    environmental_augmentation=True,
    military_environment=MilitaryEnvironment.GROUND_MOBILE
)
ml_attack.train(train_challenges, train_responses)

# Enhanced Side-Channel Attack
sca = EnhancedSideChannelAttack(
    attack_type='power',
    military_environment=MilitaryEnvironment.GROUND_MOBILE,
    em_shielding=False
)
sca.train(challenges, responses)

# Supply Chain Attack
supply_attack = SupplyChainAttack(
    tampering_rate=0.01,
    detection_difficulty=0.8
)
supply_attack.train(challenges, responses)

# Fault Injection Attack
fault_attack = FaultInjectionAttack(
    injection_type='voltage',  # voltage, clock, or laser
    precision=0.8,
    strength=0.5
)
fault_attack.train(challenges, responses)
```

## Use Cases

### Military Communication Protocol

```python
from ppet.use_cases.secure_communication import SecureCommunicationProtocol

# Initialize protocol with military environment
protocol = SecureCommunicationProtocol(
    challenge_length=64,
    military_environment=MilitaryEnvironment.GROUND_MOBILE
)

# Enroll and authenticate under stress
device_id = "military_device_001"
protocol.enroll_device(device_id)
success, confidence = protocol.authenticate_device(device_id)

# Secure communication
if success:
    key = protocol.generate_session_key(device_id)
    encrypted = protocol.encrypt_message(device_id, message)
    decrypted = protocol.decrypt_message(device_id, encrypted)
```

### Military Drone Authentication

```python
from ppet.use_cases.drone_authentication import DroneAuthenticationProtocol

# Initialize with military parameters
protocol = DroneAuthenticationProtocol(
    challenge_length=128,
    military_environment=MilitaryEnvironment.AIRCRAFT_EXTERNAL
)

# Enrolll drone with mission parameters
drone_id = "military_drone_001"
location = (37.7749, -122.4194, 1000.0)
protocol.enroll_drone(drone_id, location)

# Authenticate with environmental consideration
success, confidence, metrics = protocol.authenticate_drone(
    drone_id,
    num_auth_crps=10,
    location=(37.7750, -122.4195, 1200.0)
)
```

## Configuration

Military-grade configuration options:

```yaml
# military_config.yaml
military_environment:
  type: "ground_mobile"  # ground_mobile, aircraft_internal, aircraft_external, naval_sheltered, naval_exposed, space_vehicle
  mission_duration: 1000.0  # hours
  em_shielding: false

puf_params:
  challenge_length: 64
  environmental_sensitivity: true
  aging_enabled: true
  # Environmental parameter ranges (validated at initialization)
  temperature_range: [-65.0, 125.0]  # °C, military approximation
  voltage_range: [0.8, 1.4]          # V, military approximation
  em_noise_range: [0.0, 2.0]         # normalized units
  
# Calibrated sensitivity parameters (aligned with documentation)
environmental_sensitivity:
  temperature: 0.00005  # per °C
  voltage: 0.05         # per V²
  aging_alpha: 0.1      # maximum aging factor
  aging_tau: 8760.0     # hours (1 year time constant)

security_analysis:
  ml_attack_models: ["rf", "mlp"]
  environmental_augmentation: true
  supply_chain_tampering_rate: 0.01
  fault_injection_types: ["voltage", "clock", "laser"]

auth_threshold: 0.9
confidence_threshold: 0.85

logging:
  level: INFO
  file: ppet_military.log
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## Best Practices

1. **Military Environment Selection**
   - Choose appropriate environment profile for the application
   - Consider mission duration and environmental extremes
   - Test across full temperature and EMI ranges
   - Validate aging effects for long-term missions

2. **Enhanced Security Analysis**
   - Test against all military-grade attack vectors
   - Include environmental variations in attack models
   - Monitor PUF behavior across mission timeline
   - Validate supply chain security measures

3. **Authentication Protocol**
   - Use sufficient CRPs for military-grade security
   - Implement proper session management
   - Handle environmental variations gracefully
   - Monitor for tampering attempts

4. **Military Configuration**
   - Document all military-specific parameters
   - Version control configuration files
   - Test with different environmental profiles
   - Validate against military standards

5. **Comprehensive Testing**
   - Test under all supported military environments
   - Validate against military specifications
   - Include long-term reliability testing
   - Document all test conditions and results

## Military Standards Compliance

PPET implements models based on military approximations:

1. **Environmental Parameters**
   - Temperature cycling and extreme conditions
   - Voltage tolerance ranges
   - Electromagnetic interference modeling
   - Aging and degradation effects

2. **EMI Modeling**
   - Conducted and radiated susceptibility
   - Frequency-dependent interference
   - Pulse characteristics and timing
   - Environment-specific profiles

3. **Stress Testing**
   - Temperature cycling approximations
   - Mechanical stress modeling
   - Long-term reliability assessment
   - Mission-critical validation 