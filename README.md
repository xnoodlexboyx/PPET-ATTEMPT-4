# PPET: Physical Unclonable Function Emulation and Analysis Tool

PPET is a comprehensive framework for emulating and analyzing Physical Unclonable Functions (PUFs) in hardware security applications. The framework provides accurate PUF models, security analysis tools, and practical use cases for PUF-based authentication and secure communication.

## Features

- **PUF Emulators**
  - Arbiter PUF with realistic manufacturing variations and environmental effects
  - SRAM PUF modeling transistor mismatch
  - Ring Oscillator PUF with spatial correlation
  
- **Security Analysis**
  - Machine learning based modeling attacks
  - Side-channel attack simulation
  - Uniqueness and reliability metrics
  
- **Use Cases**
  - Secure device authentication
  - Drone-to-ground communication
  - Session key generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppet.git
cd ppet
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic PUF Emulation

```python
from ppet.core.puf_emulator import ArbiterPUF

# Create PUF instance
puf = ArbiterPUF(challenge_length=64)

# Generate Challenge-Response Pairs (CRPs)
challenges, responses = puf.generate_crps(num_crps=1000)
```

### Security Analysis

```python
from ppet.core.analysis import PUFAnalyzer
from ppet.core.threat_simulator import MLAttack

# Analyze PUF characteristics
analyzer = PUFAnalyzer()
uniqueness = analyzer.analyze_uniqueness(responses)
reliability = analyzer.analyze_reliability(responses, noise_responses)

# Simulate ML attack
attack = MLAttack(model_type='rf')
attack.train(challenges, responses)
success_rate = attack.evaluate(test_challenges, test_responses)
```

### Secure Communication Example

```python
from ppet.use_cases.secure_communication import SecureCommunicationProtocol

# Initialize protocol
protocol = SecureCommunicationProtocol(challenge_length=64)

# Enroll and authenticate device
device_id = "device_001"
protocol.enroll_device(device_id)
success, confidence = protocol.authenticate_device(device_id)

# Generate session key and exchange messages
if success:
    key = protocol.generate_session_key(device_id)
    encrypted = protocol.encrypt_message(device_id, message)
    decrypted = protocol.decrypt_message(device_id, encrypted)
```

## Configuration

PPET can be configured using YAML or JSON configuration files. Example configuration:

```yaml
puf_params:
  challenge_length: 64
  noise_sigma: 0.1
  variation_sigma: 0.2

auth_threshold: 0.9

logging:
  level: INFO
  file: ppet.log
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## Documentation

Detailed documentation is available in the `docs` directory:
- [User Guide](docs/user_guide.md)
- API Reference (coming soon)
- [Examples](examples/)

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PPET in your research, please cite:

```bibtex
@software{ppet2024,
  title = {PPET: Physical Unclonable Function Emulation and Analysis Tool},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ppet}
}
``` 