import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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
    """Machine learning based modeling attack."""
    
    def __init__(
        self,
        model_type: str = 'rf',
        model_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize ML attack.
        
        Args:
            model_type: Type of ML model ('rf' for Random Forest or 'mlp' for Neural Network)
            model_params: Optional parameters for the ML model
        """
        super().__init__(f'ML_{model_type.upper()}')
        self.model_type = model_type
        self.model_params = model_params or {}
        
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
    
    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Train ML model on CRP data.
        
        Args:
            challenges: Challenge bit vectors
            responses: Response bits
        """
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            challenges, responses,
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