from setuptools import setup, find_packages

setup(
    name="mlflow_genai-fun",
    version="1.0.0",
    author="Andre Mesarovic",
    description="MLflow GenAI Fun",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url="https://github.com/amesar/mlflow-genai-fun",
    project_urls={
        "Bug Tracker": "https://github.com/amesar/mlflow-genai-fun/issues",
        "Documentation": "https://github.com/amesar/mlflow-genai-fun/",
        "Source Code": "https://github.com/amesar/mlflow-genai-fun/"
    },
    python_requires = ">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
    install_requires=[
        "mlflow-skinny[databricks]>=2.11.2",
        "mlflow[genai]",
        "databricks-vectorsearch",
        "databricks-cli",
        "pandas>=2.0.3",
        "mdutils",
        "wheel"
    ],
    extras_require= {
        "tests": [ "mlflow", "pytest","pytest-html>=3.2.0", "shortuuid>=1.0.11" ],
        "streamlit": [ "streamlit" ]
    },
    license = "Apache License 2.0",
    keywords = "mlflow ml ai",
    classifiers = [
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ]
)
