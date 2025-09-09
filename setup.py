from setuptools import setup, find_packages

setup(
    name="ner-training",
    version="0.1.0",
    description="Named Entity Recognition Training Project",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "seqeval>=1.2.2",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
    ],
)