from setuptools import setup, find_packages

setup(
    name="gpudefrag",
    version="2.0.0",
    description="NVIDIA-Grade ML Infrastructure for GPU Memory Optimization",
    author="GPU Defrag Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=12.0",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "triton>=2.1.0",
    ],
    scripts=["run.py"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
