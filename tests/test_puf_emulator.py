import unittest
import numpy as np
from ppet.core.puf_emulator import ArbiterPUF, SRAMPUF, RingOscillatorPUF

class TestPUFEmulator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.nominal_conditions = {
            'temperature': 25.0,  # °C
            'voltage': 1.2,      # V
            'em_noise': 0.0      # normalized units
        }
        self.stress_conditions = {
            'temperature': 85.0,  # °C
            'voltage': 1.1,      # V
            'em_noise': 0.1      # normalized units
        }

    def test_arbiter_puf_uniqueness(self):
        """Test that different Arbiter PUF instances produce unique responses."""
        n_stages = 64
        num_instances = 10
        num_challenges = 1000
        
        # Create multiple PUF instances
        pufs = [
            ArbiterPUF(n_stages=n_stages, seed=i, environmental_stressors=self.nominal_conditions)
            for i in range(num_instances)
        ]
        
        # Generate responses for same challenges
        challenges = np.random.randint(0, 2, size=(num_challenges, n_stages))
        responses = []
        
        for puf in pufs:
            _, resp = puf.generate_crps(num_challenges, challenges)
            responses.append(resp)
        
        responses = np.array(responses)
        
        # Calculate Hamming distances between all pairs
        hamming_distances = []
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = np.mean(responses[i] != responses[j])
                hamming_distances.append(distance)
        
        avg_distance = np.mean(hamming_distances)
        
        # Ideal uniqueness is 0.5 (50% difference between instances)
        self.assertGreater(avg_distance, 0.45)
        self.assertLess(avg_distance, 0.55)

    def test_arbiter_puf_reliability(self):
        """Test Arbiter PUF reliability under environmental stress."""
        n_stages = 64
        num_challenges = 1000
        
        # Create PUF instance
        puf = ArbiterPUF(n_stages=n_stages, seed=self.seed, environmental_stressors=self.nominal_conditions)
        
        # Generate nominal responses
        challenges, nominal_responses = puf.generate_crps(num_challenges)
        
        # Generate responses under stress
        puf.environmental_stressors = self.stress_conditions
        _, stress_responses = puf.generate_crps(num_challenges, challenges)
        
        # Calculate bit error rate
        bit_errors = np.mean(nominal_responses != stress_responses)
        
        # Bit error rate should be low but non-zero under stress
        self.assertGreater(bit_errors, 0.0)
        self.assertLess(bit_errors, 0.15)

    def test_sram_puf_uniqueness(self):
        """Test that different SRAM PUF instances produce unique patterns."""
        rows, columns = 32, 32
        num_instances = 10
        num_reads = 10
        
        # Create multiple PUF instances
        pufs = [
            SRAMPUF(rows=rows, columns=columns, seed=i, environmental_stressors=self.nominal_conditions)
            for i in range(num_instances)
        ]
        
        # Generate startup patterns
        patterns = []
        for puf in pufs:
            responses = puf.generate_crps(num_reads)
            patterns.append(responses[0])  # Take first response as reference
        
        patterns = np.array(patterns)
        
        # Calculate Hamming distances between all pairs
        hamming_distances = []
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = np.mean(patterns[i] != patterns[j])
                hamming_distances.append(distance)
        
        avg_distance = np.mean(hamming_distances)
        
        # Ideal uniqueness is 0.5
        self.assertGreater(avg_distance, 0.45)
        self.assertLess(avg_distance, 0.55)

    def test_sram_puf_reliability(self):
        """Test SRAM PUF reliability under environmental stress."""
        rows, columns = 32, 32
        num_reads = 100
        
        # Create PUF instance
        puf = SRAMPUF(rows=rows, columns=columns, seed=self.seed, environmental_stressors=self.nominal_conditions)
        
        # Generate nominal patterns
        nominal_patterns = puf.generate_crps(num_reads)
        reference_pattern = nominal_patterns[0]
        
        # Calculate nominal stability
        nominal_errors = np.mean([np.mean(pattern != reference_pattern) for pattern in nominal_patterns[1:]])
        
        # Test under stress
        puf.environmental_stressors = self.stress_conditions
        stress_patterns = puf.generate_crps(num_reads)
        
        # Calculate stability under stress
        stress_errors = np.mean([np.mean(pattern != reference_pattern) for pattern in stress_patterns])
        
        # Bit error rates should be low but increase under stress
        self.assertLess(nominal_errors, 0.05)
        self.assertGreater(stress_errors, nominal_errors)
        self.assertLess(stress_errors, 0.15)

    def test_ro_puf_uniqueness(self):
        """Test that different RO PUF instances produce unique responses."""
        num_oscillators = 128
        num_instances = 10
        num_challenges = 1000
        
        # Create multiple PUF instances
        pufs = [
            RingOscillatorPUF(
                num_oscillators=num_oscillators,
                seed=i,
                environmental_stressors=self.nominal_conditions
            )
            for i in range(num_instances)
        ]
        
        # Generate responses for same challenges
        challenges = [
            tuple(np.random.choice(num_oscillators, 2, replace=False))
            for _ in range(num_challenges)
        ]
        
        responses = []
        for puf in pufs:
            _, resp = puf.generate_crps(num_challenges, challenges)
            responses.append(resp)
        
        responses = np.array(responses)
        
        # Calculate Hamming distances between all pairs
        hamming_distances = []
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = np.mean(responses[i] != responses[j])
                hamming_distances.append(distance)
        
        avg_distance = np.mean(hamming_distances)
        
        # Ideal uniqueness is 0.5
        self.assertGreater(avg_distance, 0.45)
        self.assertLess(avg_distance, 0.55)

    def test_ro_puf_reliability(self):
        """Test RO PUF reliability under environmental stress."""
        num_oscillators = 128
        num_challenges = 1000
        
        # Create PUF instance
        puf = RingOscillatorPUF(
            num_oscillators=num_oscillators,
            seed=self.seed,
            environmental_stressors=self.nominal_conditions
        )
        
        # Generate nominal responses
        challenges = [
            tuple(np.random.choice(num_oscillators, 2, replace=False))
            for _ in range(num_challenges)
        ]
        _, nominal_responses = puf.generate_crps(num_challenges, challenges)
        
        # Generate responses under stress
        puf.environmental_stressors = self.stress_conditions
        _, stress_responses = puf.generate_crps(num_challenges, challenges)
        
        # Calculate bit error rate
        bit_errors = np.mean(nominal_responses != stress_responses)
        
        # Bit error rate should be low but non-zero under stress
        self.assertGreater(bit_errors, 0.0)
        self.assertLess(bit_errors, 0.15)

    def test_environmental_effects(self):
        """Test that environmental conditions affect PUF behavior consistently."""
        # Test temperature effect
        temp_conditions = self.nominal_conditions.copy()
        temp_conditions['temperature'] = 85.0
        
        # Test voltage effect
        voltage_conditions = self.nominal_conditions.copy()
        voltage_conditions['voltage'] = 1.1
        
        # Test noise effect
        noise_conditions = self.nominal_conditions.copy()
        noise_conditions['em_noise'] = 0.1
        
        # Test each PUF type
        puf_configs = [
            (ArbiterPUF, {'n_stages': 64}),
            (SRAMPUF, {'rows': 32, 'columns': 32}),
            (RingOscillatorPUF, {'num_oscillators': 128})
        ]
        
        for PUFClass, params in puf_configs:
            # Create PUF instances
            puf_nominal = PUFClass(seed=self.seed, environmental_stressors=self.nominal_conditions, **params)
            puf_temp = PUFClass(seed=self.seed, environmental_stressors=temp_conditions, **params)
            puf_voltage = PUFClass(seed=self.seed, environmental_stressors=voltage_conditions, **params)
            puf_noise = PUFClass(seed=self.seed, environmental_stressors=noise_conditions, **params)
            
            # Generate responses
            if isinstance(puf_nominal, SRAMPUF):
                resp_nominal = puf_nominal.generate_crps(100)
                resp_temp = puf_temp.generate_crps(100)
                resp_voltage = puf_voltage.generate_crps(100)
                resp_noise = puf_noise.generate_crps(100)
            else:
                _, resp_nominal = puf_nominal.generate_crps(100)
                _, resp_temp = puf_temp.generate_crps(100)
                _, resp_voltage = puf_voltage.generate_crps(100)
                _, resp_noise = puf_noise.generate_crps(100)
            
            # Calculate differences from nominal
            diff_temp = np.mean(resp_nominal != resp_temp)
            diff_voltage = np.mean(resp_nominal != resp_voltage)
            diff_noise = np.mean(resp_nominal != resp_noise)
            
            # Verify that environmental changes cause measurable but not excessive differences
            self.assertGreater(diff_temp, 0.0)
            self.assertLess(diff_temp, 0.2)
            self.assertGreater(diff_voltage, 0.0)
            self.assertLess(diff_voltage, 0.2)
            self.assertGreater(diff_noise, 0.0)
            self.assertLess(diff_noise, 0.2)

if __name__ == '__main__':
    unittest.main() 