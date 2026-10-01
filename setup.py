"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.6.0",
    "mujoco-warp==3.11.0",
    "mujoco==3.11.0",
    "warp-lang==1.17.0",
    "rsl-rl-lib==5.4.2",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
