from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

requirements = [
    'setuptools',
    'torch',
    'einops',
    'flash_attn',
    'transformers',
]



setup(
    name="faesm",
    version='0.0.4',
    keywords=['LLM', 'PLM', 'protein language model'],
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",

    packages=find_packages(include=['faesm*']),
    include_package_data=True,
    zip_safe=True,

    install_requires=requirements,

    test_suite='tests',
)
