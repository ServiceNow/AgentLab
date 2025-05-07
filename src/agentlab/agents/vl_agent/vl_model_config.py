from .vl_model import LlamaModelArgs

VL_MODEL_ARGS_DICT = {
    "llama_32_11b": LlamaModelArgs(
        model_name="llama_32_11b",
        model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype="bfloat16",
        checkpoint_dir=None,
        max_length=32768,
        max_new_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    )
}
