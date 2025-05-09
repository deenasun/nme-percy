"""Environments module for reinforcement learning.

This module contains custom Gymnasium environments for reinforcement learning
navigation through grid environments derived from depth maps.
"""

from src.environments.navigation_env import NavigationEnv

__all__ = ["NavigationEnv"] 