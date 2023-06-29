from setuptools import setup, find_packages

setup(
    name="geolocation_generator",
    author="rsh",
    author_email="",
    description="Detect geolocation entities from the texts",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    # Dependencies
    install_requires=[
        "pandas",
        "numpy",
        "pandasql",
        "whoosh",
        "unidecode",
        "scikit-learn",
        "nltk",
        "spacy==3.4.0",
        "typing-extensions==4.5.0"
    ],

    extras_require = {
        'transformers':  ["transformers"]
    },
    
    entry_points={
        "console_scripts": [
            "download-pkgs = geolocation_generator:download_pkgs",
        ]
    },

    version="0.1",
    license="MIT",
    long_description=open("README.md").read(),
)