import numpy as np
from ppet.use_cases.secure_communication import SecureCommunicationProtocol

def main():
    """Demonstrate secure communication protocol."""
    # Initialize protocol
    protocol = SecureCommunicationProtocol(
        challenge_length=64,
        num_crps=1000
    )
    
    # Enroll device
    device_id = "device_001"
    enrollment_data = protocol.enroll_device(device_id)
    print("\nDevice Enrollment:")
    print(f"Enrolled device {device_id} with {enrollment_data['num_crps']} CRPs")
    
    # Authenticate device
    success, confidence = protocol.authenticate_device(device_id)
    print("\nDevice Authentication:")
    print(f"Authentication {'successful' if success else 'failed'}")
    print(f"Confidence: {confidence:.2f}")
    
    if success:
        # Generate session key
        key = protocol.generate_session_key(device_id, key_length=256)
        if key is None:
            print("Failed to generate session key")
            return
            
        print("\nSession Key Generation:")
        print(f"Generated {len(key)}-bit session key")
        
        # Example message exchange
        message = np.random.randint(2, size=256)  # Random binary message
        print("\nMessage Exchange:")
        print(f"Original message: {message[:10]}...")
        
        # Encrypt message
        encrypted = protocol.encrypt_message(device_id, message)
        if encrypted is None:
            print("Failed to encrypt message")
            return
            
        print(f"Encrypted message: {encrypted[:10]}...")
        
        # Decrypt message
        decrypted = protocol.decrypt_message(device_id, encrypted)
        if decrypted is None:
            print("Failed to decrypt message")
            return
            
        print(f"Decrypted message: {decrypted[:10]}...")
        
        # Verify successful decryption
        if np.array_equal(message, decrypted):
            print("Message successfully encrypted and decrypted!")
        else:
            print("Error: Decryption failed!")

if __name__ == "__main__":
    main() 