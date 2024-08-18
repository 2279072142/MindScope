import tiktoken
from typing import List, Dict


def count_message_tokens(messages : List[Dict[str, str]], model : str = "gpt-3.5-turbo-16k") -> int:
    """
    Returns the number of tokens used by a list of messages.

    Args:
    messages (list): A list of messages, each of which is a dictionary containing the role and content of the message.
    model (str): The name of the model to use for tokenization. Defaults to "gpt-3.5-turbo-0301".

    Returns:
    int: The number of tokens used by the list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # !Node: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return count_message_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # !Note: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return count_message_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-16k":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def count_dollar(input_token_num:int,output_token_num:int,model_name:str= "gpt-3.5-turbo-16k") ->float:
    '''
    :param token_num:   token花费数量
    :param model_name:   模型名称
    :return cost:      美刀花费
    '''
    input_cost=0
    output_cost=0
    if model_name=="gpt-3.5-turbo-16k":
        input_cost=input_token_num/1000*0.003
        output_cost=output_token_num/1000*0.004
    elif model_name=="gpt-3.5-turbo":
        input_cost = input_token_num / 1000 * 0.0015
        output_cost = output_token_num / 1000 * 0.002
    elif model_name=="gpt4":
        input_cost = input_token_num / 1000 * 0.03
        output_cost = output_token_num / 1000 * 0.06

    return input_cost+output_cost


def count_string_tokens(string: str, model_name: str = "gpt-3.5-turbo-16k") -> int:
    """
    Returns the number of tokens in a text string.

    Args:
    string (str): The text string.
    model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
