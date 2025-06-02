from setuptools import setup, find_packages

setup(
    name='sca_test',
    version='0.1.0',
    description='Smart Composite Augmentation with automatic transformation search',
    author='Anna Kolesnikova',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'opencv-python',
        'albumentations',
        'optuna',
        'numpy',
        'scikit-learn'
    ],
    python_requires='>=3.7',
)

