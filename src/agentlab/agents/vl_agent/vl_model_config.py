from .vl_model import VLModelArgs

VL_MODEL_ARGS_DICT = {
    "llama_32_11b": VLModelArgs(
        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=100_000,
        vision_support=False,
    )
}
