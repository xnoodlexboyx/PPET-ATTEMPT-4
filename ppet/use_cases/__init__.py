"""Use cases module containing defense-specific PUF applications."""

from .secure_communication import SecureCommunicationUseCase
from .drone_authentication import DroneAuthenticationUseCase

__all__ = [
    'SecureCommunicationUseCase',
    'DroneAuthenticationUseCase'
] 