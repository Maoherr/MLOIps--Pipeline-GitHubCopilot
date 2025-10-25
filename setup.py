from setuptools import setup, find_packages

setup(
    name="mlops_pipeline",
    version="0.1.0",
    description="Pipeline ML reproducible",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)