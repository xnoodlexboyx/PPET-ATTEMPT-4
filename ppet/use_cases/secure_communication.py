import numpy as np
from typing import Tuple, Dict, Optional
from ..core.puf_emulator import ArbiterPUF
from ..utilities.config_manager import load_config
from ..utilities.logging import setup_logger

logger = setup_logger(__name__)

class SecureCommunicationProtocol:
    """PUF-based secure communication protocol implementation."""
    
    def __init__(
        self,
        challenge_length: int = 64,
        num_crps: int = 1000,
        config_path: Optional[str] = None
    ):
        """Initialize secure communication protocol.
        
        Args:
            challenge_length: Length of PUF challenge bits
            num_crps: Number of CRPs to use for authentication
            config_path: Optional path to configuration file
        """
        self.challenge_length = challenge_length
        self.num_crps = num_crps
        
        # Load configuration if provided
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize PUFs for both parties
        self.device_puf = ArbiterPUF(
            challenge_length=challenge_length,
            **self.config.get('puf_params', {})
        )
        
        # Initialize CRP database
        self.crp_db = {}
        self.session_keys = {}
    
    def enroll_device(self, device_id: str) -> Dict:
        """Enroll a device by generating and storing CRPs.
        
        Args:
            device_id: Unique identifier for the device
        
        Returns:
            Enrollment data including CRP database
        """
        logger.info(f"Enrolling device: {device_id}")
        
        # Generate CRPs
        challenges, responses = self.device_puf.generate_crps(self.num_crps)
        
        # Store in database
        self.crp_db[device_id] = {
            'challenges': challenges,
            'responses': responses,
            'used_crps': set()
        }
        
        logger.info(f"Device {device_id} enrolled with {self.num_crps} CRPs")
        
        return {
            'device_id': device_id,
            'num_crps': self.num_crps,
            'challenge_length': self.challenge_length
        }
    
    def authenticate_device(
        self,
        device_id: str,
        num_auth_crps: int = 10
    ) -> Tuple[bool, float]:
        """Authenticate a device using stored CRPs.
        
        Args:
            device_id: Device identifier
            num_auth_crps: Number of CRPs to use for authentication
        
        Returns:
            Authentication success and confidence score
        """
        if device_id not in self.crp_db:
            logger.error(f"Device {device_id} not enrolled")
            return False, 0.0
        
        device_data = self.crp_db[device_id]
        unused_indices = list(
            set(range(self.num_crps)) - device_data['used_crps']
        )
        
        if len(unused_indices) < num_auth_crps:
            logger.error(f"Insufficient unused CRPs for device {device_id}")
            return False, 0.0
        
        # Select random unused CRPs
        auth_indices = np.random.choice(
            unused_indices,
            size=num_auth_crps,
            replace=False
        )
        
        # Get challenges and expected responses
        auth_challenges = device_data['challenges'][auth_indices]
        expected_responses = device_data['responses'][auth_indices]
        
        # Get actual responses from device
        actual_responses = self.device_puf.generate_responses(auth_challenges)
        
        # Calculate match rate
        matches = np.sum(actual_responses == expected_responses)
        confidence = matches / num_auth_crps
        
        # Mark CRPs as used
        device_data['used_crps'].update(auth_indices)
        
        # Authentication successful if confidence above threshold
        success = confidence >= self.config.get('auth_threshold', 0.9)
        logger.info(
            f"Device {device_id} authentication: "
            f"{'success' if success else 'failed'} "
            f"(confidence: {confidence:.2f})"
        )
        
        return success, confidence
    
    def generate_session_key(
        self,
        device_id: str,
        key_length: int = 256
    ) -> Optional[np.ndarray]:
        """Generate session key using PUF responses.
        
        Args:
            device_id: Device identifier
            key_length: Length of session key in bits
        
        Returns:
            Session key bit array if successful, None otherwise
        """
        # Authenticate device first
        success, confidence = self.authenticate_device(device_id)
        if not success:
            logger.error(
                f"Key generation failed - device {device_id} "
                "authentication failed"
            )
            return None
        
        # Generate new challenges for key generation
        key_challenges = np.random.randint(
            2,
            size=(key_length, self.challenge_length)
        )
        
        # Get responses from device
        key_responses = self.device_puf.generate_responses(key_challenges)
        
        # Store session key
        self.session_keys[device_id] = key_responses
        
        logger.info(
            f"Generated {key_length}-bit session key for device {device_id}"
        )
        return key_responses
    
    def encrypt_message(
        self,
        device_id: str,
        message: np.ndarray
    ) -> Optional[np.ndarray]:
        """Encrypt message using session key.
        
        Args:
            device_id: Device identifier
            message: Binary message to encrypt
        
        Returns:
            Encrypted message if successful, None otherwise
        """
        if device_id not in self.session_keys:
            logger.error(f"No session key found for device {device_id}")
            return None
        
        key = self.session_keys[device_id]
        if len(message) != len(key):
            logger.error("Message length must match key length")
            return None
        
        # Simple XOR encryption
        encrypted = np.bitwise_xor(message, key)
        return encrypted
    
    def decrypt_message(
        self,
        device_id: str,
        encrypted_message: np.ndarray
    ) -> Optional[np.ndarray]:
        """Decrypt message using session key.
        
        Args:
            device_id: Device identifier
            encrypted_message: Encrypted binary message
        
        Returns:
            Decrypted message if successful, None otherwise
        """
        if device_id not in self.session_keys:
            logger.error(f"No session key found for device {device_id}")
            return None
        
        key = self.session_keys[device_id]
        if len(encrypted_message) != len(key):
            logger.error("Message length must match key length")
            return None
        
        # XOR decryption (same as encryption)
        decrypted = np.bitwise_xor(encrypted_message, key)
        return decrypted 