from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

requirements = [
    "setuptools",
    "torch",
    "einops",
    "transformers",
]

extras_require = {
    "flash_attn": ["flash_attn"],
}

# requirements for testing
extras_require["test"] = [
    "pytest",
    "numpy",
    "biopython",
    "matplotlib",
    "seaborn",
]
setup(
    name="faesm",
    version="0.1.0",
    keywords=["LLM", "PLM", "protein language model"],
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["faesm*"]),
    include_package_data=True,
    zip_safe=True,
    install_requires=requirements,
    extras_require=extras_require,
    test_suite="tests",
)
