# Databricks notebook source
# MAGIC %md
# MAGIC # LLM RAG Evaluation with MLflow using llama2-as-judge Example Notebook
# MAGIC
# MAGIC In this notebook, we will demonstrate how to evaluate various a RAG system with MLflow. We will use llama2-70b as the judge model, via a Databricks serving endpoint.
# MAGIC
# MAGIC <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/rag-evaluation-llama2.ipynb" class="notebook-download-btn"><i class="fas fa-download"></i>Download this Notebook</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manifest
# MAGIC
# MAGIC #### Source
# MAGIC
# MAGIC * This notebook is a slight adaptation of the [RAG Evaluation Notebook (using llama2-70b-as-judge)](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/index.html#rag-evaluation-notebook-using-llama2-70b-as-judge)
# MAGIC from the MLflow doc site
# MAGIC * https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/llm-evaluate/notebooks/rag-evaluation-llama2.ipynb
# MAGIC * Clone date: 2024-04-05
# MAGIC
# MAGIC #### Changes
# MAGIC * Added `!pip install chromadb`
# MAGIC
# MAGIC #### Status - Errors
# MAGIC
# MAGIC * InvalidRequestError: The model `text-davinci-003` has been deprecated, learn more here: https://platform.openai.com/docs/
# MAGIC deprecations
# MAGIC * https://platform.openai.com/docs/deprecations
# MAGIC * recommended replacement: gpt-3.5-turbo-instruct
# MAGIC * https://github.com/mlflow/mlflow/blob/master/mlflow/utils/openai_utils.py#L114

# COMMAND ----------

!pip install chromadb
!pip install mlflow-skinny==2.11.3
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("1. Secrets scope", "dbdemos")
scope = dbutils.widgets.get("1. Secrets scope")

dbutils.widgets.text("2. Secrets key", "openai-key")
key = dbutils.widgets.get("2. Secrets key")

scope, key

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set OpenAI Key

# COMMAND ----------

import os
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=scope, key=key)

# COMMAND ----------

# MAGIC %md
# MAGIC We need to set our OpenAI API key.
# MAGIC
# MAGIC In order to set your private key safely, please be sure to either export your key through a command-line terminal for your current instance, or, for a permanent addition to all user-based sessions, configure your favored environment management configuration file (i.e., .bashrc, .zshrc) to have the following entry:
# MAGIC
# MAGIC `OPENAI_API_KEY=<your openai API key>`

# COMMAND ----------

#os.environ["DATABRICKS_HOST"] = "REDACTED"
#os.environ["DATABRICKS_TOKEN"] = "REDACTED"

# COMMAND ----------

import pandas as pd
import mlflow
import langchain
import openai

print("mlflow.version:   ", mlflow.__version__)
print("langchain.version:", langchain.__version__)
print("openai.version:   ", openai.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC Set the deployment target to "databricks" for use with Databricks served models.

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a RAG system
# MAGIC
# MAGIC Use Langchain and Chroma to create a RAG system that answers questions based on the MLflow documentation.

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# COMMAND ----------

loader = WebBaseLoader("https://mlflow.org/docs/latest/index.html")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the RAG system using `mlflow.evaluate()`

# COMMAND ----------

# MAGIC %md
# MAGIC Create a simple function that runs each input through the RAG chain

# COMMAND ----------

def model(input_df):
    answer = []
    for index, row in input_df.iterrows():
        answer.append(qa(row["questions"]))
    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC Create an eval dataset

# COMMAND ----------

eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "How to run mlflow.evaluate()?",
            "How to log_table()?",
            "How to load_table()?",
        ],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a faithfulness metric using `databricks-llama2-70b-chat` as the judge

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, faithfulness

# Create a good and bad example for faithfulness in the context of this problem
faithfulness_examples = [
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. ",
        score=2,
        justification="The output provides a working solution, using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions.",
        score=5,
        justification="The output provides a solution that is using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
]

faithfulness_metric = faithfulness(
    model="endpoints:/databricks-llama-2-70b-chat", examples=faithfulness_examples
)
print(faithfulness_metric)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a relevance metric using `databricks-llama2-70b-chat` as the judge

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, relevance

relevance_metric = relevance(model="endpoints:/databricks-llama-2-70b-chat")
print(relevance_metric)

# COMMAND ----------

results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "source_documents",
        }
    },
)
print(results.metrics)

# COMMAND ----------

results.tables["eval_results_table"]
