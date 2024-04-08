# Databricks notebook source
# MAGIC %md ### MLflow LLM Evaluation Notebooks
# MAGIC
# MAGIC #### Overview
# MAGIC * Databricks notebooks for MLflow LLM evaluation examples.
# MAGIC * Notebooks adapted from Databricks and MLflow web sites.
# MAGIC * Note: ntermittent OpenAI rate limit errors prevent succesfull execution of notebooks.
# MAGIC
# MAGIC #### Notebooks
# MAGIC * [question-answering-evaluation]($question-answering-evaluation)
# MAGIC * [rag-evaluation]($rag-evaluation)
# MAGIC * [rag-evaluation-llama2]($rag-evaluation-llama2)
# MAGIC
# MAGIC #### Original downloaded notebooks
# MAGIC * [Original notebooks]($original) - as downloaded on 2024-04-05
# MAGIC
# MAGIC ###### Last updated _2024-04-08_

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Documention
# MAGIC
# MAGIC ##### Databricks Documention
# MAGIC   * [LLM evaluation with MLflow example notebook](https://docs.databricks.com/_extras/notebooks/source/mlflow/question-answering-evaluation.html) - link to notebook - question-answering-evaluation.dbc
# MAGIC     * [Evaluate large language models with MLflow](https://docs.databricks.com/en/mlflow/llm-evaluate.html) - general article
# MAGIC
# MAGIC ##### MLflow Documention
# MAGIC [LLM Evaluation Examples](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#llm-evaluation-examples) - Jupyter notebook download links
# MAGIC * [QA Evaluation Notebook](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#qa-evaluation-notebook) - question-answering-evaluation.ipynb
# MAGIC * [RAG Evaluation Notebook (using gpt-4-as-judge)](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#rag-evaluation-notebook-using-gpt-4-as-judge) - rag-evaluation.ipynb
# MAGIC * [RAG Evaluation Notebook (using llama2-70b-as-judge)
# MAGIC ](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#rag-evaluation-notebook-using-llama2-70b-as-judge) - rag-evaluation-llama2.ipynb
# MAGIC * [Evaluating a 🤗 Hugging Face LLM Notebook (using gpt-4-as-judge)](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#evaluating-a-hugging-face-llm-notebook-using-gpt-4-as-judge) - huggingface-evaluation.ipynb
# MAGIC
# MAGIC More evaluation info
# MAGIC * [Guides and Tutorials for LLM Model Evaluation](https://mlflow.org/docs/latest/model-evaluation/index.html#guides-and-tutorials-for-llm-model-evaluation)
# MAGIC   * [LLM Evaluation with MLflow Example Notebook](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/question-answering-evaluation.html)
# MAGIC * [mlflow.evaluate](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate)
