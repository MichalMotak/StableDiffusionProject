from setuptools import setup, find_packages

setup(
    name='surgrid',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your package dependencies here
        # e.g., 'requests>=2.23.0',
    ],
    author='Ssharvien Kumar',
    description='SurGrID: Controllable Surgical Simulation via Scene Graph to Image Diffusion',
)