import typing
from documenthelper.va import load_llama_model_configs

class TestVA:
    def test_load_configs_correct_keys(self) -> None:
        expected_keys = ["llama_cpp_shared_configs", "llama_embeddings_configs", "llama_llm_configs"]
        configs = load_llama_model_configs()
        assert all([k in configs for k in expected_keys]), "Missing keys from config!"
        
        

