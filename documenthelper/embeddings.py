"""Embeddings module.

This module is responsible for adding document embeddings in our local VectorStore.
"""
# Python generic imports
import typing  # pylint: disable=unused-import

# import argparse
from collections import ChainMap

# Langchain imports
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# local repo imports
from documenthelper.utils import load_llama_model_configs, load_vectorstore


def prepare_http_content(
    url: str, chunk_size: int = 512, chunk_overlap: int = 0
) -> list[Document]:
    """Reads a webpage and returns document chunks to encode and add into a VectorStore.

    Args:
        url (str): The url to load content from.
        chunk_size (int, optional): Max chunk size. Defaults to 512.
        chunk_overlap (int, optional): How much should chunks overlap. Defaults to 0.

    Returns:
        list[Document]: The webpage content split into a list of chunks.
    """
    # Read new document in
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def main() -> None:
    """Main function is used as entrypoint for `poetry run embeddings` command."""
    # TODO: remove hardcoded url when we extend to add embeddings from various sources.
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    # Create argparser and get url if given
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--url", type=str, default=url)
    # args = parser.parse_args()
    # print(args)

    # Load configs
    config = load_llama_model_configs()

    # Instantiate embeddings to use to transform documents to vectors before storing
    embeddings = LlamaCppEmbeddings(
        **ChainMap(
            # since we are only loading embeddings we can afford loading the model on VRAM
            {"n_gpu_layers": 35},
            config["llama_embeddings_configs"],
            config["llama_cpp_shared_configs"],
        )
    )

    # Instatiate Chroma store
    vectorstore = load_vectorstore(embeddings)
    document_chunks = prepare_http_content(url)

    vectorstore.add_documents(document_chunks)
    vectorstore.persist()

    print("New document(s) stored successfully! Exiting.")


if __name__ == "__main__":
    main()
