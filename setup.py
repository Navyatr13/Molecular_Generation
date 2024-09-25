from setuptools import setup, find_packages

setup(
    name='vae_project',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',  # Add other dependencies like numpy, pandas, tqdm, etc.
        'numpy',
        'pandas',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'main=main:main',  # Maps 'train_vae' to the main function in main.py
        ],
    },
)
