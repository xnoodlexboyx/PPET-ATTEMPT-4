import numpy as np
from typing import Dict, Optional, Tuple
from ..core.puf_emulator import ArbiterPUF
from ..utilities.config_manager import load_config
from ..utilities.logging import setup_logger

logger = setup_logger(__name__)

class DroneAuthenticationProtocol:
    """PUF-based drone authentication protocol implementation."""
    
    def __init__(
        self,
        challenge_length: int = 128,
        num_crps: int = 1000,
        config_path: Optional[str] = None
    ):
        """Initialize drone authentication protocol.
        
        Args:
            challenge_length: Length of PUF challenge bits
            num_crps: Number of CRPs to use for authentication
            config_path: Optional path to configuration file
        """
        self.challenge_length = challenge_length
        self.num_crps = num_crps
        
        # Load configuration if provided
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize drone PUF
        self.drone_puf = ArbiterPUF(
            challenge_length=challenge_length,
            **self.config.get('puf_params', {})
        )
        
        # Initialize database for enrolled drones
        self.enrolled_drones = {}
        self.active_sessions = {}
    
    def enroll_drone(
        self,
        drone_id: str,
        location: Optional[Tuple[float, float, float]] = None
    ) -> Dict:
        """Enroll a drone by generating and storing CRPs.
        
        Args:
            drone_id: Unique identifier for the drone
            location: Optional (latitude, longitude, altitude) tuple
        
        Returns:
            Enrollment data including CRP database
        """
        logger.info(f"Enrolling drone: {drone_id}")
        
        # Generate CRPs
        challenges, responses = self.drone_puf.generate_crps(self.num_crps)
        
        # Store enrollment data
        self.enrolled_drones[drone_id] = {
            'challenges': challenges,
            'responses': responses,
            'used_crps': set(),
            'location': location,
            'auth_attempts': 0,
            'last_auth_time': None
        }
        
        logger.info(f"Drone {drone_id} enrolled with {self.num_crps} CRPs")
        
        return {
            'drone_id': drone_id,
            'num_crps': self.num_crps,
            'challenge_length': self.challenge_length,
            'location': location
        }
    
    def authenticate_drone(
        self,
        drone_id: str,
        num_auth_crps: int = 10,
        location: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[bool, float, Dict]:
        """Authenticate a drone using stored CRPs.
        
        Args:
            drone_id: Drone identifier
            num_auth_crps: Number of CRPs to use for authentication
            location: Optional current (latitude, longitude, altitude)
        
        Returns:
            (success, confidence, metrics) tuple
        """
        if drone_id not in self.enrolled_drones:
            logger.error(f"Drone {drone_id} not enrolled")
            return False, 0.0, {}
        
        drone_data = self.enrolled_drones[drone_id]
        unused_indices = list(
            set(range(self.num_crps)) - drone_data['used_crps']
        )
        
        if len(unused_indices) < num_auth_crps:
            logger.error(f"Insufficient unused CRPs for drone {drone_id}")
            return False, 0.0, {}
        
        # Select random unused CRPs
        auth_indices = np.random.choice(
            unused_indices,
            size=num_auth_crps,
            replace=False
        )
        
        # Get challenges and expected responses
        auth_challenges = drone_data['challenges'][auth_indices]
        expected_responses = drone_data['responses'][auth_indices]
        
        # Get actual responses from drone
        actual_responses = self.drone_puf.generate_responses(auth_challenges)
        
        # Calculate match rate
        matches = np.sum(actual_responses == expected_responses)
        confidence = matches / num_auth_crps
        
        # Update drone data
        drone_data['used_crps'].update(auth_indices)
        drone_data['auth_attempts'] += 1
        drone_data['last_auth_time'] = np.datetime64('now')
        
        if location:
            drone_data['location'] = location
        
        # Authentication successful if confidence above threshold
        success = confidence >= self.config.get('auth_threshold', 0.9)
        
        # Prepare metrics
        metrics = {
            'auth_attempts': drone_data['auth_attempts'],
            'remaining_crps': self.num_crps - len(drone_data['used_crps']),
            'location': drone_data['location'],
            'last_auth_time': drone_data['last_auth_time']
        }
        
        logger.info(
            f"Drone {drone_id} authentication: "
            f"{'success' if success else 'failed'} "
            f"(confidence: {confidence:.2f})"
        )
        
        return success, confidence, metrics
    
    def establish_secure_channel(
        self,
        drone_id: str,
        session_key_length: int = 256
    ) -> Optional[Dict]:
        """Establish secure communication channel with authenticated drone.
        
        Args:
            drone_id: Drone identifier
            session_key_length: Length of session key in bits
        
        Returns:
            Session data if successful, None otherwise
        """
        # Authenticate drone first
        success, confidence, _ = self.authenticate_drone(drone_id)
        if not success:
            logger.error(
                f"Secure channel establishment failed - "
                f"drone {drone_id} authentication failed"
            )
            return None
        
        # Generate session key using PUF responses
        key_challenges = np.random.randint(
            2,
            size=(session_key_length, self.challenge_length)
        )
        key_responses = self.drone_puf.generate_responses(key_challenges)
        
        # Create session
        session_id = np.random.bytes(16).hex()
        session_data = {
            'session_id': session_id,
            'key': key_responses,
            'start_time': np.datetime64('now'),
            'drone_id': drone_id
        }
        
        self.active_sessions[session_id] = session_data
        
        logger.info(
            f"Established secure channel with drone {drone_id} "
            f"(session: {session_id})"
        )
        
        return session_data
    
    def verify_session(self, session_id: str) -> bool:
        """Verify if a session is valid.
        
        Args:
            session_id: Session ID to verify
        
        Returns:
            True if session is valid
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session_age = (
            np.datetime64('now') - session['start_time']
        ) / np.timedelta64(1, 'h')
        
        # Session expires after 24 hours
        if session_age > 24:
            self.end_session(session_id)
            return False
        
        return True
    
    def end_session(self, session_id: str) -> None:
        """End a secure communication session.
        
        Args:
            session_id: Session ID to end
        """
        if session_id in self.active_sessions:
            drone_id = self.active_sessions[session_id]['drone_id']
            del self.active_sessions[session_id]
            logger.info(
                f"Ended secure channel session {session_id} "
                f"with drone {drone_id}"
            )
    
    def refresh_crp_database(self, drone_id: str) -> bool:
        """Refresh CRP database for a drone when running low.
        
        Args:
            drone_id: Drone identifier
        
        Returns:
            True if refresh successful
        """
        if drone_id not in self.enrolled_drones:
            logger.error(f"Drone {drone_id} not enrolled")
            return False
        
        drone_data = self.enrolled_drones[drone_id]
        if len(drone_data['used_crps']) > 0.8 * self.num_crps:
            # Generate new CRPs
            challenges, responses = self.drone_puf.generate_crps(self.num_crps)
            
            # Update drone data
            drone_data['challenges'] = challenges
            drone_data['responses'] = responses
            drone_data['used_crps'].clear()
            
            logger.info(f"Refreshed CRP database for drone {drone_id}")
            return True
        
        return False 