# Here we'll try to create a Virtual Assistant, parsing a question and 
import os
import typing
from collections import ChainMap
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_llama_model_configs() -> dict:
    """Prepare llama models' configs, both for Embeddings and LLM.
    Returns:
        dict - contains configs for llama models.
    """
    # TODO: move configs to config file
    llama_cpp_shared_configs = {
        "n_gpu_layers": 35,                                 # max number of layers to get offloaded to GPU (35 according to model's metadata)
        "n_batch": 512,                                     # Tokens to process in parallel
        "n_ctx": 2048,                                      # Context window length (should be 4096 for llama2)
        "f16_kv": True                                      # lower precision for less mem consumption
    }

    llama_embeddings_configs = {
        "model_path": "../models/llama-2-q4_0.gguf",
        "n_gpu_layers": 1
    }

    llama_llm_configs={
        "model_path": "../models/llama-2-chat-q4_0.gguf", 
        "temperature":0,
        "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
        "verbose": True,
    }
    return {"llama_cpp_shared_configs": llama_cpp_shared_configs, "llama_embeddings_configs": llama_embeddings_configs, "llama_llm_configs": llama_llm_configs}

def load_vectorstore(embeddings: LlamaCppEmbeddings, vectorstore_path: str="../vectorstore/") -> Chroma:
    """Load a locally stored Chroma vectorstore containing our embedded documents.
    Args:
        embeddings (langchain.embeddings.LlamaCppEmbeddings): The embeddings model to use when creating the questions embeddings.
        vectorstore_path (str): The path to the local vectorstore, defaults to "../vectorstore/".

    Returns:
        langchain.vectorstores.Chroma - vectorstore instance to use.
    """
    # TODO: move configs to config file
    # Load vectorstore from disk
    if os.listdir(vectorstore_path):
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=vectorstore_path
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
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.<</SYS>> 
        Question: {question} 
        Context: {context} 
        Answer:[/INST]
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return qa_chain


def main():
    # LLama2 model configs
    # Some configs taken from https://python.langchain.com/docs/guides/local_llms#llamacpp
    llama_model_configs = load_llama_model_configs()
 
    # Instantiate embeddings to use to transform documents to vectors before storing
    embeddings = LlamaCppEmbeddings(**ChainMap(llama_model_configs["llama_embeddings_configs"],
                                               llama_model_configs["llama_cpp_shared_configs"]))

    # Instantiate llama llm
    llm = LlamaCpp(**ChainMap(llama_model_configs["llama_llm_configs"], llama_model_configs["llama_cpp_shared_configs"]))

    vectorstore = load_vectorstore(embeddings)

    qa_chain = prepare_qa_chain(llm, vectorstore)
    
    while True:
        question = input("Ask a question:\n")
        qa_chain({"query": question})

if __name__ == "__main__":
    main()
