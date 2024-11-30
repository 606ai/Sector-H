from setuptools import setup, find_packages

setup(
    name="universe-a",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'PyQt6>=6.6.1',
        'requests>=2.31.0',
        'transformers>=4.36.2',
        'torch>=2.1.2',
        'langchain>=0.1.0',
        'python-dotenv>=1.0.0',
        'huggingface-hub>=0.20.1'
    ],
    author="Universe-A Team",
    author_email="admin@aiworlds.icu",
    description="An AI-driven universe simulation platform",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://aiworlds.icu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'universe-a=universe_a.main:main',
        ],
    },
)
