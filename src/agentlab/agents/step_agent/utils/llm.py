import os
import re
import copy
# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     AutoModelForSeq2SeqLM,
# )
# import ctranslate2
from time import sleep
import openai
import tiktoken

import warnings
warnings.simplefilter("ignore")

input_token_cost_usd_by_model = {
    "gpt-4-1106-preview": 0.01 / 1000,
    "gpt-4": 0.03 / 1000,
    "gpt-4-32k": 0.06 / 1000,
    "gpt-3.5-turbo": 0.001 / 1000,
    "gpt-3.5-turbo-instruct": 0.0015 / 1000,
    "gpt-3.5-turbo-16k": 0.003 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

output_token_cost_usd_by_model = {
    "gpt-4-1106-preview": 0.03 / 1000,
    "gpt-4": 0.06 / 1000,
    "gpt-4-32k": 0.12 / 1000,
    "gpt-3.5-turbo": 0.002 / 1000,
    "gpt-3.5-turbo-instruct": 0.002 / 1000,
    "gpt-3.5-turbo-16k": 0.004 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

def fill_prompt_template(prompt_template, objective, observation, url, previous_history):
    prompt = copy.deepcopy(prompt_template)
    print(observation, type(observation))
    prompt["input"] = prompt["input"].replace("{objective}", objective)
    prompt["input"] = prompt["input"].replace("{observation}", observation)
    prompt["input"] = prompt["input"].replace("{url}", url)   
    prompt["input"] = prompt["input"].replace("{previous_actions}", previous_history)   
    return prompt

def filter_quotes_if_matches_template(action):
    if action is None:
        return None

    # Regex pattern to match the entire 'type [X] ["Y"]' template, allowing for Y to be digits as well
    pattern = r'^type \[\d+\] \["([^"\[\]]+)"\]$'
    # Check if the action matches the specific template
    match = re.match(pattern, action)
    if match:
        # Extract the matched part that needs to be unquoted
        y_part = match.group(1)
        # Reconstruct the action string without quotes around Y
        filtered_action = f'type [{match.group(0).split("[")[1].split("]")[0]}] [{y_part}]'
        return filtered_action
    else:
        # Return the original action if it doesn't match the template
        return action

def parse_action_reason(model_response):
    reason_match = re.search(r'REASON:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
    reason = reason_match.group(1) if reason_match else None

    action_match = re.search(r'ACTION:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
    action = action_match.group(1) if action_match else None
    
    action = filter_quotes_if_matches_template(action)
    
    return action, reason

# def construct_llm_message_hf(prompt, prompt_mode, model_type="llama2"):
#     if model_type == "llama2":
#         instruction = "<s>[INST] " + prompt["instruction"]
#     else:
#         instruction = prompt["instruction"]
    
#     messages = [{"role": "system", "content": instruction}]
    
#     if prompt["examples"]:
#         messages.append({"role": "system", "content": "Here are a few examples:"})
#         for example in prompt["examples"]:
#             messages.append({"role": "system", "content": f"\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"})
    
#     if model_type == "llama2":
#         query = f"\nHere is the current Input. Please respond with REASON and ACTION.\n### Input:\n{prompt['input']}\n[/INST]\n"
#     else:
#         query = f"\nHere is the current Input. Please respond with REASON and ACTION.\n### Input:\n{prompt['input']}\n\n### Response:"
    
#     messages.append({"role": "user", "content": query})
#     if prompt_mode == "chat":
#         return messages
#     elif prompt_mode == "completion":
#         all_content = ''.join(message['content'] for message in messages)
#         messages_completion = [{"role": "user", "content": all_content}]
#         return messages_completion
    

def construct_llm_message_openai(prompt, prompt_mode):
    messages = [{"role": "system", "content": prompt["instruction"]}]
        
    if prompt["examples"]:
        messages.append({"role": "system", "content": "Here are a few examples:"})
        for example in prompt["examples"]:
            messages.append({"role": "system", "content": f"\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"})
    
    messages.append({"role": "user", "content": f"Here is the current Input. Please respond with REASON and ACTION.\n### Input:\n{prompt['input']}\n\n### Response:"})
    if prompt_mode == "chat":
        return messages
    elif prompt_mode == "completion":
        all_content = ''.join(message['content'] for message in messages)
        messages_completion = [{"role": "user", "content": all_content}]
        return messages_completion

def call_openai_llm(messages, model="gpt-3.5-turbo", **model_kwargs):
    """
    Sends a request with a chat conversation to OpenAI's chat API and returns a response.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to send to the chatbot.
        model (str)
            The model to use for the chatbot. Default is "gpt-3.5-turbo".
        temperature (float)
            The temperature to use for the chatbot. Defaults to 0. Note that a temperature
            of 0 does not guarantee the same response (https://platform.openai.com/docs/models/gpt-3-5).
    
    Returns:
        response (Optional[dict])
            The response from OpenAI's chat API, if any.
    """
    client = openai.AzureOpenAI(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
    temperature = model_kwargs.get('temperature', 0.3)
    top_p = model_kwargs.get('top_p', 1.0)
    n = model_kwargs.get('n', 1)
    
    num_attempts = 0
    while True:
        try:
            if model=="text-davinci-003":
                response = client.chat.completions.create(
                model=model,
                prompt=messages[0]["content"],
                temperature=temperature,
                top_p=top_p,
                n=n,
                max_tokens=128)
                return response.choices[0].text.strip()
                        
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=n
            )
            return response.choices[0].message.content.strip()
        except openai.error.AuthenticationError as e:
            print(e)
            return None
        except openai.error.RateLimitError as e:
            print(e)
            print("Sleeping for 10 seconds...")
            sleep(10)
            num_attempts += 1
        except openai.error.ServiceUnavailableError as e:
            print(e)
            print("Sleeping for 10 seconds...")
            sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10 seconds...")
            sleep(10)
            num_attempts += 1

def get_num_tokens(text: str, model_name: str) -> int:
    tokenizer = tiktoken.encoding_for_model(model_name=model_name)
    return len(tokenizer.encode_ordinary(text))

def calculate_cost_openai(messages: str, response: str, model_name: str) -> int:
    input_text = " ".join([msg["content"] for msg in messages]) 
    num_input_tokens = get_num_tokens(input_text, model_name)
    num_output_tokens = get_num_tokens(response, model_name)
    
    input_token_cost = input_token_cost_usd_by_model.get(model_name, None)
    output_token_cost = output_token_cost_usd_by_model.get(model_name, None)
    if input_token_cost is None or output_token_cost is None:
        print(f"[calculate_cost_openai] unknown model {model_name}")
        return 0
    return num_input_tokens * input_token_cost + num_output_tokens * output_token_cost

# def load_tokenizer(mpath, context_size):
#     tokenizer = AutoTokenizer.from_pretrained(mpath, return_token_type_ids=False)
#     # tokenizer.pad_token = tokenizer.eos_token
#     # tokenizer.pad_token_id = tokenizer.eos_token_id
#     # tokenizer.model_max_length = context_size
#     # tokenizer.padding_side = "right"
#     # tokenizer.truncation_side = "left"
#     # tokenizer.add_eos_token = True
#     return tokenizer

# def load_model(mpath, dtype, device="cuda", context_len=4096, is_seq2seq=False, ct2_mpath=None):
#     if is_seq2seq:
#         model_loader = AutoModelForSeq2SeqLM
#     else:
#         model_loader = AutoModelForCausalLM

#     if dtype == "bf16":
#         model = model_loader.from_pretrained(
#             mpath,
#             max_position_embeddings=context_len,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch.bfloat16,
#             device_map="balanced_low_0",
#         )
#     elif dtype == "4bit":
#         model = model_loader.from_pretrained(
#             mpath,
#             max_position_embeddings=context_len,
#             low_cpu_mem_usage=True,
#             load_in_4bit=True,
#             device_map="auto",
#         )
#     elif dtype == "4bit-optimized":
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )
#         model = model_loader.from_pretrained(
#             mpath,
#             use_cache=True,
#             device_map="auto",
#             quantization_config=bnb_config,
#             max_position_embeddings=context_len,
#         )
#     elif dtype == "8bit":
#         model = model_loader.from_pretrained(
#             mpath,
#             max_position_embeddings=context_len,
#             low_cpu_mem_usage=True,
#             load_in_8bit=True,
#             device_map="auto",
#         )
#     elif dtype == "ct2":
#         assert ct2_mpath is not None
#         model = ctranslate2.Generator(ct2_mpath, device=device)

#     return model

# @torch.no_grad()
# def generate_prediction(
#     inputs,
#     model,
#     tokenizer,
#     max_new_tokens,
#     is_seq2seq=False,
#     **kwargs,
#     # num_beams,
#     # do_sample,
#     # no_repeat_ngram_size,
#     # temperature,
#     # top_k,
#     # top_p,
# ):
#     input_ids = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_new_tokens).input_ids
    
#     outputs = model.generate(
#         input_ids=input_ids.cuda(),
#         max_new_tokens=max_new_tokens,
#         **kwargs,
#     ).cpu()

#     torch.cuda.empty_cache()
#     if not is_seq2seq:
#         outputs = outputs[:, input_ids.shape[1] :]

#     prediction = [
#         p.split(tokenizer.pad_token, 1)[0]
#         for p in tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     ][0].strip()
        
#     return prediction

# def generate_prediction(
#     inputs,
#     model,
#     tokenizer,
#     **kwargs,
# ):
#     inputs = tokenizer([inputs], return_tensors='pt', truncation=True, add_special_tokens=False).to(model.device)
    
#     # if torch.cuda.is_available():
#     #     inputs = inputs.to('cuda')
#     outputs = model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         **kwargs,
#     )
    
#     outputs = outputs[:, inputs.input_ids.shape[1] :]
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#     return prediction