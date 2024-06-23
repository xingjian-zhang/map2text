from setuptools import setup, find_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="llm4explore",
    version="0.1",
    packages=find_packages(),
    install_requires=load_requirements(),
)
