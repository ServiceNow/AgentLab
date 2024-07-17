import os
import re
from typing import Union
import copy
from time import sleep
import openai
import tiktoken

from langchain_openai import AzureChatOpenAI, ChatOpenAI

import warnings
warnings.simplefilter("ignore")


def fill_prompt_template(prompt_template: dict[str, str], objective: str, observation: str, url: str, previous_history: str) -> dict[str, str]:
    prompt = copy.deepcopy(prompt_template)
    prompt["input"] = prompt["input"].replace("{objective}", objective)
    prompt["input"] = prompt["input"].replace("{observation}", observation)
    prompt["input"] = prompt["input"].replace("{url}", url)   
    prompt["input"] = prompt["input"].replace("{previous_actions}", previous_history)   
    return prompt

def filter_quotes_if_matches_template(action: Union[str, None]) -> str:
    if action is None:
        return

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

def parse_action_reason(model_response: str) -> tuple[str, str]:
    reason_match = re.search(r'REASON:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
    reason = reason_match.group(1) if reason_match else None

    action_match = re.search(r'ACTION:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
    action = action_match.group(1) if action_match else None
    
    action = filter_quotes_if_matches_template(action)
    
    return action, reason
    

def construct_llm_message_openai(prompt: str, prompt_mode: str):
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

def call_openai_llm(messages: list[dict[str, str]], model: Union[ChatOpenAI, AzureChatOpenAI], **model_kwargs) -> str:
    """
    Sends a request with a chat conversation to OpenAI's chat API and returns a response.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to send to the chatbot.
        model (ChatOpenAI | AzureChatOpenAI)
            The model to use for the chatbot. Default is "gpt-3.5-turbo".
        temperature (float)
            The temperature to use for the chatbot. Defaults to 0. Note that a temperature
            of 0 does not guarantee the same response (https://platform.openai.com/docs/models/gpt-3-5).
    
    Returns:
        response (str)
            The response from OpenAI's chat API, if any.
    """
    
    num_attempts = 0
    while True:
        try:
            model_response = model.invoke(messages)
            return model_response.content.strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10 seconds...")
            sleep(10)
            num_attempts += 1
        except openai.APIStatusError as e:
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
