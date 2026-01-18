from setuptools import setup, find_packages

setup(
    name="liquid_mamba_kan",
    version="0.1.0",
    description="A hybrid deep learning architecture combining Liquid Networks, Mamba, and KANs.",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
)
