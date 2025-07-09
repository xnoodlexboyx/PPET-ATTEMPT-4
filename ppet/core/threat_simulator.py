import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from .military_stressors import MilitaryStressors, MilitaryEnvironment

class Attack:
    """Base class for PUF attacks."""
    
    def __init__(self, name: str):
        """Initialize attack.
        
        Args:
            name: Name of the attack
        """
        self.name = name
        self.success_rate = 0.0
        self.model = None
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Train attack model on CRP data.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        raise NotImplementedError
    
    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """Predict responses for given challenges.
        
        Args:
            challenges: Challenge bit vectors
        
        Returns:
            Predicted response bits
        """
        raise NotImplementedError
    
    def evaluate(self, challenges: np.ndarray, true_responses: np.ndarray) -> float:
        """Evaluate attack success rate.
        
        Args:
            challenges: Challenge bit vectors
            true_responses: True response bits
        
        Returns:
            Attack success rate (0.0 to 1.0)
        """
        pred_responses = self.predict(challenges)
        self.success_rate = accuracy_score(true_responses, pred_responses)
        return self.success_rate

class MLAttack(Attack):
    """Enhanced machine learning based modeling attack with military considerations."""
    
    def __init__(
        self,
        model_type: str = 'rf',
        model_params: Optional[Dict[str, Any]] = None,
        environmental_augmentation: bool = False,
        military_environment: Optional[MilitaryEnvironment] = None
    ):
        """Initialize ML attack.
        
        Args:
            model_type: Type of ML model ('rf' for Random Forest or 'mlp' for Neural Network)
            model_params: Optional parameters for the ML model
            environmental_augmentation: Whether to augment training data with environmental variations
            military_environment: Optional military environment for environmental augmentation
        """
        super().__init__(f'ML_{model_type.upper()}_{"ENV" if environmental_augmentation else "STD"}')
        self.model_type = model_type
        self.model_params = model_params or {}
        self.environmental_augmentation = environmental_augmentation
        self.military_environment = military_environment
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                **self.model_params
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        if environmental_augmentation and military_environment:
            self.stressor = MilitaryStressors(environment=military_environment)
    
    def _augment_data(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        augmentation_factor: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data with environmental variations.
        
        Args:
            challenges: Original challenge bit vectors
            responses: Original response bits
            augmentation_factor: How many times to augment each sample
            
        Returns:
            Augmented challenges and responses
        """
        if not self.environmental_augmentation or not self.military_environment:
            return challenges, responses
            
        num_samples = len(challenges)
        aug_challenges = np.tile(challenges, (augmentation_factor, 1))
        aug_responses = np.tile(responses, augmentation_factor)
        
        # Add environmental feature columns
        times = np.random.uniform(0, 1000, num_samples * augmentation_factor)
        env_features = np.zeros((num_samples * augmentation_factor, 3))
        
        for i, time in enumerate(times):
            stressors = self.stressor.get_all_stressors(time)
            env_features[i] = [
                stressors['temperature'],
                stressors['em_noise'],
                stressors['aging_factor']
            ]
        
        # Combine challenge bits with environmental features
        aug_challenges = np.hstack([aug_challenges, env_features])
        
        return aug_challenges, aug_responses
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Train ML model on CRP data with environmental augmentation.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        # Augment data if enabled
        X, y = self._augment_data(challenges, responses)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        print(f"{self.name} validation accuracy: {val_accuracy:.4f}")
    
    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """Predict responses using trained ML model.
        
        Args:
            challenges: Challenge bit vectors
        
        Returns:
            Predicted response bits
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(challenges)

class SideChannelAttack(Attack):
    """Power/timing side-channel attack simulation."""
    
    def __init__(
        self,
        attack_type: str = 'power',
        noise_std: float = 0.1,
        num_measurements: int = 100
    ):
        """Initialize side-channel attack.
        
        Args:
            attack_type: Type of side-channel ('power' or 'timing')
            noise_std: Standard deviation of measurement noise
            num_measurements: Number of measurements per challenge
        """
        super().__init__(f'SCA_{attack_type.upper()}')
        self.attack_type = attack_type
        self.noise_std = noise_std
        self.num_measurements = num_measurements
        self.stage_weights = None
        self.threshold = None
    
    def _simulate_measurements(
        self,
        challenges: np.ndarray,
        stage_delays: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate side-channel measurements.
        
        Args:
            challenges: Challenge bit vectors
            stage_delays: Optional true stage delays for simulation
        
        Returns:
            Simulated measurements
        """
        num_challenges = len(challenges)
        num_stages = challenges.shape[1]
        
        if stage_delays is None:
            # Generate random stage delays if not provided
            stage_delays = np.random.normal(1.0, 0.1, size=num_stages)
        
        # Base measurements without noise
        measurements = np.zeros((num_challenges, self.num_measurements))
        
        for i in range(num_challenges):
            # Calculate path delays
            path_delay = np.sum(stage_delays * challenges[i])
            
            if self.attack_type == 'power':
                # Power consumption model: quadratic with path delay
                base_measurement = path_delay ** 2
            else:  # timing
                # Timing model: linear with path delay
                base_measurement = path_delay
            
            # Add measurement noise
            measurements[i] = base_measurement + np.random.normal(
                0,
                self.noise_std,
                size=self.num_measurements
            )
        
        return measurements
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Train side-channel attack model.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        # Simulate measurements for training data
        measurements = self._simulate_measurements(challenges)
        
        # Average measurements to reduce noise
        avg_measurements = np.mean(measurements, axis=1)
        
        # Find optimal threshold
        sorted_measurements = np.sort(avg_measurements)
        thresholds = (sorted_measurements[1:] + sorted_measurements[:-1]) / 2
        best_accuracy = 0
        best_threshold = None
        
        for threshold in thresholds:
            pred = (avg_measurements > threshold).astype(int)
            accuracy = accuracy_score(responses, pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"{self.name} training accuracy: {best_accuracy:.4f}")
    
    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """Predict responses using side-channel measurements.
        
        Args:
            challenges: Challenge bit vectors
        
        Returns:
            Predicted response bits
        """
        if self.threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Simulate and process measurements
        measurements = self._simulate_measurements(challenges)
        avg_measurements = np.mean(measurements, axis=1)
        
        # Classify based on threshold
        return (avg_measurements > self.threshold).astype(int) 

class EnhancedSideChannelAttack(SideChannelAttack):
    """Enhanced side-channel attack with military-grade capabilities."""
    
    def __init__(
        self,
        attack_type: str = 'power',
        noise_std: float = 0.1,
        num_measurements: int = 100,
        military_environment: Optional[MilitaryEnvironment] = None,
        em_shielding: bool = False
    ):
        """Initialize enhanced side-channel attack.
        
        Args:
            attack_type: Type of side-channel ('power', 'timing', or 'em')
            noise_std: Standard deviation of measurement noise
            num_measurements: Number of measurements per challenge
            military_environment: Optional military environment profile
            em_shielding: Whether EM shielding is present
        """
        super().__init__(attack_type, noise_std, num_measurements)
        self.military_environment = military_environment
        self.em_shielding = em_shielding
        
        if military_environment:
            self.stressor = MilitaryStressors(environment=military_environment)
    
    def _simulate_measurements(
        self,
        challenges: np.ndarray,
        stage_delays: Optional[np.ndarray] = None,
        time: float = 0.0
    ) -> np.ndarray:
        """Simulate enhanced side-channel measurements.
        
        Args:
            challenges: Challenge bit vectors
            stage_delays: Optional true stage delays for simulation
            time: Mission time for environmental effects
            
        Returns:
            Simulated measurements
        """
        measurements = super()._simulate_measurements(challenges, stage_delays)
        
        if self.military_environment:
            # Get environmental conditions
            stressors = self.stressor.get_all_stressors(time)
            
            # Apply environmental effects
            if self.attack_type == 'power':
                # Temperature affects power consumption
                temp_effect = 1.0 + 0.002 * (stressors['temperature'] - 25.0)
                measurements *= temp_effect
            
            # EMI effects
            if not self.em_shielding:
                emi_noise = np.random.normal(
                    0,
                    stressors['em_noise'] * self.noise_std,
                    size=measurements.shape
                )
                measurements += emi_noise
        
        return measurements

class SupplyChainAttack(Attack):
    """Supply chain attack simulation for military hardware."""
    
    def __init__(
        self,
        tampering_rate: float = 0.01,  # 1% of components tampered
        detection_difficulty: float = 0.8  # 80% difficulty to detect
    ):
        """Initialize supply chain attack.
        
        Args:
            tampering_rate: Rate of component tampering
            detection_difficulty: Difficulty of detecting tampering
        """
        super().__init__('SUPPLY_CHAIN')
        self.tampering_rate = tampering_rate
        self.detection_difficulty = detection_difficulty
        self.tampered_indices = None
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Simulate supply chain attack training.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        num_components = challenges.shape[1]
        num_tampered = int(num_components * self.tampering_rate)
        
        # Randomly select components to tamper
        self.tampered_indices = np.random.choice(
            num_components,
            size=num_tampered,
            replace=False
        )
    
    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """Predict responses for tampered PUF.
        
        Args:
            challenges: Challenge bit vectors
            
        Returns:
            Predicted response bits
        """
        if self.tampered_indices is None:
            raise RuntimeError("Attack not trained. Call train() first.")
            
        # Flip challenge bits for tampered components
        tampered_challenges = challenges.copy()
        tampered_challenges[:, self.tampered_indices] = 1 - tampered_challenges[:, self.tampered_indices]
        
        # Simulate tampered responses
        responses = np.zeros(len(challenges))
        for i, challenge in enumerate(tampered_challenges):
            # Simple response generation for tampered PUF
            responses[i] = np.sum(challenge) % 2
            
        return responses

class FaultInjectionAttack(Attack):
    """Fault injection attack simulation."""
    
    def __init__(
        self,
        injection_type: str = 'voltage',  # voltage, clock, or laser
        precision: float = 0.8,  # Injection timing precision
        strength: float = 0.5  # Normalized injection strength
    ):
        """Initialize fault injection attack.
        
        Args:
            injection_type: Type of fault injection
            precision: Timing precision of injection (0-1)
            strength: Strength of injection (0-1)
        """
        super().__init__(f'FAULT_{injection_type.upper()}')
        self.injection_type = injection_type
        self.precision = precision
        self.strength = strength
        self.vulnerable_stages = None
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Identify vulnerable stages for fault injection.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        num_stages = challenges.shape[1]
        stage_sensitivity = np.zeros(num_stages)
        
        # Analyze stage sensitivity
        for i in range(num_stages):
            mod_challenges = challenges.copy()
            mod_challenges[:, i] = 1 - mod_challenges[:, i]
            
            # Calculate response changes
            response_changes = np.sum(responses != (np.sum(mod_challenges, axis=1) % 2))
            stage_sensitivity[i] = response_changes / len(responses)
        
        # Select most vulnerable stages
        threshold = np.percentile(stage_sensitivity, 80)  # Top 20% sensitive stages
        self.vulnerable_stages = np.where(stage_sensitivity >= threshold)[0]
    
    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """Predict responses under fault injection.
        
        Args:
            challenges: Challenge bit vectors
            
        Returns:
            Predicted response bits
        """
        if self.vulnerable_stages is None:
            raise RuntimeError("Attack not trained. Call train() first.")
            
        responses = np.zeros(len(challenges))
        
        for i, challenge in enumerate(challenges):
            # Apply fault to vulnerable stages
            faulted_challenge = challenge.copy()
            
            # Injection success probability based on precision
            if np.random.random() < self.precision:
                # Select random vulnerable stage
                target_stage = np.random.choice(self.vulnerable_stages)
                
                if self.injection_type == 'voltage':
                    # Voltage glitch: probabilistic bit flip
                    if np.random.random() < self.strength:
                        faulted_challenge[target_stage] = 1 - faulted_challenge[target_stage]
                elif self.injection_type == 'clock':
                    # Clock glitch: stuck at previous value
                    if target_stage > 0:
                        faulted_challenge[target_stage] = faulted_challenge[target_stage - 1]
                else:  # laser
                    # Laser fault: forced to 1
                    faulted_challenge[target_stage] = 1
            
            responses[i] = np.sum(faulted_challenge) % 2
            
        return responses 