from setuptools import setup, find_packages

setup(
    name="spikereport",
    version="0.1.0",
    description="A tool to generate PDF quality reports from Kilosort 4 results.",
    author="James",
    packages=find_packages(),
    install_requires=[
        "spikeinterface[full]>=0.100.0",  # Installs full suite including extractors
        "probeinterface",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)