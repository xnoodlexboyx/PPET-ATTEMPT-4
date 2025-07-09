# PPET User Guide

This guide provides detailed instructions for using the Physical Unclonable Function Emulation and Analysis Tool (PPET).

## Table of Contents

1. [Introduction](#introduction)
2. [PUF Emulation](#puf-emulation)
3. [Security Analysis](#security-analysis)
4. [Use Cases](#use-cases)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)

## Introduction

PPET is designed to help researchers and developers work with Physical Unclonable Functions (PUFs) by providing:
- Accurate PUF emulation with realistic manufacturing and environmental variations
- Comprehensive security analysis tools
- Practical use case implementations
- Extensible framework for custom applications

## PUF Emulation

### Arbiter PUF

The Arbiter PUF emulator models a delay-based PUF with configurable parameters:

```python
from ppet.core.puf_emulator import ArbiterPUF

# Create PUF with default parameters
puf = ArbiterPUF(challenge_length=64)

# Create PUF with custom parameters
puf = ArbiterPUF(
    challenge_length=128,
    noise_sigma=0.1,        # Environmental noise
    variation_sigma=0.2,    # Manufacturing variation
    temperature_coeff=0.01  # Temperature sensitivity
)

# Generate CRPs
challenges, responses = puf.generate_crps(
    num_crps=1000,
    temperature=25.0  # Optional environmental conditions
)
```

### SRAM PUF

The SRAM PUF emulator models memory cell behavior:

```python
from ppet.core.puf_emulator import SRAMPUF

# Create SRAM PUF
puf = SRAMPUF(
    array_size=(128, 128),  # Memory array dimensions
    mismatch_sigma=0.2      # Transistor mismatch variation
)

# Generate startup values
responses = puf.generate_responses(
    voltage=1.2,      # Supply voltage
    temperature=25.0  # Temperature
)
```

### Ring Oscillator PUF

The Ring Oscillator PUF emulator includes spatial correlation:

```python
from ppet.core.puf_emulator import RingOscillatorPUF

# Create RO-PUF
puf = RingOscillatorPUF(
    num_oscillators=256,
    correlation_length=5.0  # Spatial correlation parameter
)

# Generate frequency measurements
frequencies = puf.measure_frequencies()

# Generate responses by comparing oscillator pairs
responses = puf.generate_responses(num_pairs=100)
```

## Security Analysis

### PUF Quality Metrics

Analyze PUF characteristics using the analyzer:

```python
from ppet.core.analysis import PUFAnalyzer

analyzer = PUFAnalyzer()

# Calculate metrics
uniqueness = analyzer.analyze_uniqueness(responses)
reliability = analyzer.analyze_reliability(responses, noise_responses)
bit_aliasing = analyzer.analyze_bit_aliasing(responses)
entropy = analyzer.analyze_entropy(responses)

# Generate comprehensive report
report = analyzer.generate_report()

# Visualize metrics
analyzer.plot_metrics(save_path='puf_metrics.png')
```

### Threat Simulation

Evaluate PUF security against various attacks:

```python
from ppet.core.threat_simulator import MLAttack, SideChannelAttack

# Machine Learning Attack
ml_attack = MLAttack(model_type='rf')  # Random Forest model
ml_attack.train(train_challenges, train_responses)
success_rate = ml_attack.evaluate(test_challenges, test_responses)

# Side-Channel Attack
sca = SideChannelAttack(
    attack_type='power',
    noise_std=0.1,
    num_measurements=100
)
sca.train(challenges, responses)
predictions = sca.predict(new_challenges)
```

## Use Cases

### Secure Communication

Implement secure device authentication and communication:

```python
from ppet.use_cases.secure_communication import SecureCommunicationProtocol

# Initialize protocol
protocol = SecureCommunicationProtocol(
    challenge_length=64,
    num_crps=1000
)

# Device enrollment
device_id = "device_001"
protocol.enroll_device(device_id)

# Authentication
success, confidence = protocol.authenticate_device(device_id)

# Secure communication
if success:
    # Generate session key
    key = protocol.generate_session_key(device_id)
    
    # Exchange messages
    encrypted = protocol.encrypt_message(device_id, message)
    decrypted = protocol.decrypt_message(device_id, encrypted)
```

### Drone Authentication

Secure drone-to-ground communication:

```python
from ppet.use_cases.drone_authentication import DroneAuthenticationProtocol

# Initialize protocol
protocol = DroneAuthenticationProtocol(
    challenge_length=128,
    num_crps=1000
)

# Enroll drone with location
drone_id = "drone_001"
location = (37.7749, -122.4194, 100.0)  # lat, lon, altitude
protocol.enroll_drone(drone_id, location)

# Authenticate drone
success, confidence, metrics = protocol.authenticate_drone(
    drone_id,
    num_auth_crps=10,
    location=(37.7750, -122.4195, 120.0)
)

# Establish secure channel
if success:
    session = protocol.establish_secure_channel(drone_id)
```

## Configuration

PPET can be configured using YAML or JSON files:

```yaml
# config.yaml
puf_params:
  challenge_length: 64
  noise_sigma: 0.1
  variation_sigma: 0.2
  temperature_coeff: 0.01

auth_threshold: 0.9

logging:
  level: INFO
  file: ppet.log
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

Load configuration:

```python
from ppet.utilities.config_manager import load_config

config = load_config('config.yaml')
protocol = SecureCommunicationProtocol(config_path='config.yaml')
```

## Best Practices

1. **PUF Selection**
   - Choose PUF type based on application requirements
   - Consider resource constraints and security needs
   - Test with realistic environmental conditions

2. **Security Analysis**
   - Regularly evaluate PUF quality metrics
   - Test against multiple attack vectors
   - Monitor for changes in PUF behavior

3. **Authentication Protocol**
   - Use sufficient number of CRPs
   - Implement proper session management
   - Handle error cases gracefully

4. **Configuration**
   - Use version control for config files
   - Document parameter choices
   - Test with different configurations

5. **Testing**
   - Write comprehensive unit tests
   - Include integration tests
   - Test edge cases and error conditions 