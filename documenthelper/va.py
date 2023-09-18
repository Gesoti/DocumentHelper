"""This module is responsible for creating a Virtual Assistant for querying documents loaded in our VectorStore."""  #
# Python generic imports
import os
import typing  # pylint: disable=unused-import
import json
from collections import ChainMap

# Langchain imports
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# local repo imports
from documenthelper.utils import load_llama_model_configs


def load_vectorstore(
    embeddings: LlamaCppEmbeddings, vectorstore_path: str = "../vectorstore/"
) -> Chroma:
    """Load a locally stored Chroma vectorstore containing our embedded documents.
    Args:
        embeddings (LlamaCppEmbeddings): The embeddings model to use when creating the questions embeddings.
        vectorstore_path (str): The path to the local vectorstore, defaults to "../vectorstore/".

    Returns:
        langchain.vectorstores.Chroma - vectorstore instance to use.
    """
    # TODO: move configs to config file
    # Load vectorstore from disk
    if os.listdir(vectorstore_path):
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=vectorstore_path
        )
    return vectorstore


def prepare_qa_chain(llm: LlamaCpp, vectorstore: Chroma) -> RetrievalQA:
    """Prepares a QA chain to enable us to ask questions and get answers from an LLM.
    Args:
        llm (langchain.llms.LlamaCpp): The LLM model to use in the QA Chain
        vectorstore (langchain.vectorstores.Chroma)

    Returns:
        langchain.chains.RetrievalQA - The QA chain to converse with.
    """
    # Configure the QA chain so we can ask the LLM to answer questions based on our vectorstore's contents.
    template = """
        [INST]<<SYS>> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved Context to answer the Question.
        If the Context doesn't contain information to answer the Question, just say that you don't know.
        Use three sentences maximum and keep the Answer concise.<</SYS>>
        Question: {question}
        Context: {context}
        Answer:[/INST]
    """

    qa_chain_prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_chain_prompt},
        return_source_documents=True,
    )
    return qa_chain


def main():
    """Main method to be used as entry point for `poetry run va` command."""

    # LLama2 model configs
    # Some configs taken from https://python.langchain.com/docs/guides/local_llms#llamacpp
    llama_model_configs = load_llama_model_configs()

    # Instantiate embeddings to use to transform documents to vectors before storing
    embeddings = LlamaCppEmbeddings(
        **ChainMap(
            llama_model_configs["llama_embeddings_configs"],
            llama_model_configs["llama_cpp_shared_configs"],
        )
    )

    # Instantiate llama llm
    llm = LlamaCpp(
        **ChainMap(
            llama_model_configs["llama_llm_configs"],
            llama_model_configs["llama_cpp_shared_configs"],
        )
    )

    vectorstore = load_vectorstore(embeddings)

    qa_chain = prepare_qa_chain(llm, vectorstore)

    while True:
        question = input("Ask me a question:\n")
        qa_chain({"query": question})


if __name__ == "__main__":
    main()
