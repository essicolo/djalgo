from setuptools import setup, find_packages

setup(
    name="djalgo",
    version="0.1-alpha",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'plotly',
    ],
    # Metadata
    author="Essi Parent",
    author_email="3pp0qk4a@duck.com",
    description="A midi music composition toolkit",
    license="GPL-3",
    keywords="midi, music",
    url="https://github.com.com/essicolo/djalgo",
)