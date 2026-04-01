from setuptools import setup, find_packages # Imports the setuptools library functions: 
# - setup(): the main function that defines package metadata and dependencies
# - find_packages(): automatically discovers Python packages in the project directory

setup(
    name="av_simulation", # Package name used for installation (pip install av_simulation)
    version="0.1.0",
    description="VLA-MAC Fleet Coordination Simulation",
    packages=find_packages(),# Auto-discovers packages like 'av_simulation' and its subpackages(coordination/, control/, etc.)
    python_requires=">=3.9",
    install_requires=[
        "numpy", # Numerical computing: arrays, linear algebra, random numbers
        "Pillow", # Python Imaging Library fork: image processing for camera/vision data
        "requests", # HTTP library: used by VLM engine to communicate with Ollama/LLaVA API endpoints
        "casadi",    # Symbolic framework for numerical optimization: used by MPC (Model Predictive Control)
        "do-mpc",
        "metadrive-simulator",# MetaDrive driving simulator: provides the base environment, vehicles, and rendering
        "panda3d", # Panda3D game engine: MetaDrive's rendering backend; handles 3D visualization
    ],
)
