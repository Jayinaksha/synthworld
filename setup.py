from setuptools import setup, find_packages

setup(
    name="synthworld",
    version="0.1.0",
    description="Open-World Simulation Sandbox for Robotics Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SynthWorld Team",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pybullet>=3.2.0",
        "panda3d>=1.10.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
        "pygame>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "llm": [
            "google-generativeai>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synthworld=synthworld.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
