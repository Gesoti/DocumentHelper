"""This module is responsible for keeping functions shared between the other modules of the package."""
# Generic python imports
import typing  # pylint: disable=unused-import
import json
import os

# Langchain specific imports
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma


def read_json(filename: str) -> dict:
    """Read json files in with consistent read config.
    Args:
        filename (str): the filename (and path) of the json file to read.

    Returns:
        dict - contents of the file to read.
    """
    with open(filename, mode="r", encoding="utf-8") as file:
        contents = json.load(file)
    return contents


def load_llama_model_configs() -> dict:
    """Prepare llama models' configs, both for Embeddings and LLM.
    Returns:
        dict - contains configs for llama models.
    """
    # LLama 2 model shared configs
    llama_cpp_shared_configs = read_json("model_config/llama_shared.json")

    # LLama 2 chat model configs
    llama_llm_configs = read_json("model_config/llama_chat-config.json")
    llama_llm_configs["callback_manager"] = CallbackManager(
        [StreamingStdOutCallbackHandler()]
    )

    # LLama 2 base model configs
    llama_embeddings_configs = read_json("model_config/llama-embedding-config.json")

    return {
        "llama_cpp_shared_configs": llama_cpp_shared_configs,
        "llama_embeddings_configs": llama_embeddings_configs,
        "llama_llm_configs": llama_llm_configs,
    }


def load_vectorstore(
    embeddings: Embeddings, vectorstore_path: str = "../vectorstore/"
) -> Chroma:
    """Load a locally stored Chroma vectorstore containing our embedded documents.
    Args:
        embeddings (Embeddings): The embeddings model to use when creating the questions embeddings.
        vectorstore_path (str): The path to the local vectorstore, defaults to "../vectorstore/".

    Returns:
        Chroma - vectorstore instance to use.
    """
    # Load vectorstore from disk
    if os.listdir(vectorstore_path):
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=vectorstore_path
        )
    return vectorstore
