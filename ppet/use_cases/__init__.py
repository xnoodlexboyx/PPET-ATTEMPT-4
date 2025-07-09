"""Use cases module containing defense-specific PUF applications."""

from .secure_communication import SecureCommunicationProtocol
from .drone_authentication import DroneAuthenticationProtocol

__all__ = [
    'SecureCommunicationProtocol',
    'DroneAuthenticationProtocol'
] 