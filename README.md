# DocumentHelper
Document-reading virtual assistant

## Summary
This Virtual assistant will help you parse your own documents into a vectorstore and use LLama2 LLM to ask questions about them.

## Technical section

### How to use
*TODO*

### Preparing the llama2 models locally
1. Request access and download models using this [repo](https://github.com/facebookresearch/llama)
    1. Request access to models.
    1. You'll need to clone locally
    1. `pip install -e .` the repo
    1. Run `bash download.sh` and use the url that you received in your email after your access request has been approved.
        1. You might get errors running download.sh with `\\r`. Convert line endings to match your system.
        1. I'd suggest download the 7B or 7B-chat only initially cause it's going to take a lot of space to get them all.
1. Clone and build this [repo](https://github.com/ggerganov/llama.cpp) to transform and quantize models 
    1. Get the repo and `make` it locally.
    1. I had cuda toolkit so i used the section for cuBLAS and `make` didn't work for me so I used `CMAKE`.
        ```
        mkdir build
        cd build
        cmake .. -DLLAMA_CUBLAS=ON
        cmake --build . --config Release
        ```
    1. Then I followed instructions for [Prepare Data & Run](https://github.com/ggerganov/llama.cpp#prepare-data--run)
        1. In llama.cpp repo copy the models you've downloaded from llama repo.
        1. Check the line in that section `ls ./models` to get an idea of what you need to copy over.
        1. NOTE: `./convert` is in the root directory but `./quantize` is under `./build/bin` (it took me some time to find it)
        1. You can run inference straight from this repo but next step describes how it's used with langchain.
1. After setting up your langchain libraries in python etc, use the following [tutorial](https://python.langchain.com/docs/integrations/llms/llamacpp#gpu) to install `llama-cpp-python` and use the quantized models to run the model locally.
    1. Install `llama-cpp-python` with GPU support (for me it was cuBLAS)
    `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`
    1. Instatiate LlamaCpp model 
    `model_path="$HOME/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin"`  
    The home models here have this extension bin and gguf etc to show which ones should be used but the name doesn't matter. Just use the output of your `./quantize` script and it should work.

## Acknowledgements
Following [tutorial](https://python.langchain.com/docs/use_cases/question_answering/) from langchain website.