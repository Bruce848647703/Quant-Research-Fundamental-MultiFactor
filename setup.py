from setuptools import setup, find_packages

setup(
    name="quant-research-fundamental-multifactor",
    version="0.1.0",
    description="A股量化研究项目 - 基本面多因子月度轮动回测",
    author="Bruce",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "akshare>=1.10.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "pyarrow>=10.0.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
)
