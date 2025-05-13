import openai
from openai import OpenAI
import tiktoken
import time
import os
import re


enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

openai_api_key_name = ""
openai_api_base = ""

def GPT_response(messages, model_name):
  token_num_count = 0
  for item in messages:
    token_num_count += len(enc.encode(item["content"]))

  if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'deepseek-chat', 'gpt-4o', 'gpt-4o-2024-08-06']:
    #print(f'-------------------Model name: {model_name}-------------------')
    client = OpenAI(api_key=openai_api_key_name, base_url=openai_api_base)

    try:
      result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature = 0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except:
      try:
        result = client.chat.completions.create(
          model=model_name,
          messages=messages,
          temperature=0.0,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
      except:
        try:
          print(f'{model_name} Waiting 60 seconds for API query')
          time.sleep(60)
          result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature = 0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
          )
        except:
          return 'Out of tokens', token_num_count
    token_num_count += len(enc.encode(result.choices[0].message.content))
    print(f'Token_num_count: {token_num_count}')
    return result.choices[0].message.content, token_num_count

  else:
    raise ValueError(f'Invalid model name: {model_name}')

def GPT_response_samples(messages, model_name):
  token_num_count = 0
  result = []
  for item in messages:
    token_num_count += len(enc.encode(item["content"]))

  if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'deepseek-chat']:
    #print(f'-------------------Model name: {model_name}-------------------')
    client = OpenAI(api_key=openai_api_key_name, base_url=openai_api_base)

    try:
      results = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=1,
        temperature = 0.6,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.5
      )
    except:
      try:
        results = client.chat.completions.create(
          model=model_name,
          messages=messages,
          n=1,
          temperature=0.6,
          top_p=0.9,
          frequency_penalty=0.3,
          presence_penalty=0.5
        )
      except:
        try:
          print(f'{model_name} Waiting 60 seconds for API query')
          time.sleep(60)
          results = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1,
            temperature = 0.6,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.5
          )
        except:
          return 'Out of tokens', token_num_count

    for i in range(1):
      result.append(results.choices[i].message.content)
      token_num_count += len(enc.encode(result))

    print(f'Token_num_count: {token_num_count}')
    return result, token_num_count

  else:
    raise ValueError(f'Invalid model name: {model_name}')
  
def GPT_response_1(messages, model_name):
  token_num_count = 0
  for item in messages:
    token_num_count += len(enc.encode(item["content"]))

  if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'deepseek-chat']:
    #print(f'-------------------Model name: {model_name}-------------------')
    client = OpenAI(api_key=openai_api_key_name, base_url=openai_api_base)

    try:
      results = client.chat.completions.create(
        max_tokens=1024, 
        model=model_name,
        messages=messages,
        n=1,
        temperature = 0.6,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.5
      )
    except:
      try:
        results = client.chat.completions.create(
          max_tokens=1024,
          model=model_name,
          messages=messages,
          n=1,
          temperature=0.6,
          top_p=0.9,
          frequency_penalty=0.3,
          presence_penalty=0.5
        )
      except:
        try:
          print(f'{model_name} Waiting 60 seconds for API query')
          time.sleep(60)
          results = client.chat.completions.create(
            max_tokens=1024,
            model=model_name,
            messages=messages,
            n=1,
            temperature = 0.6,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.5
          )
        except:
          return 'Out of tokens', token_num_count

    token_num_count += len(enc.encode(results.choices[0].message.content))

    print(f'Token_num_count: {token_num_count}')
    return results.choices[0].message.content, token_num_count

  else:
    raise ValueError(f'Invalid model name: {model_name}')