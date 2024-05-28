import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Function get number of tokens"""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_content_evenly(content, parts):
    """Function split content as parts"""
    
    if parts <= 0:
        raise ValueError("Number of parts must be greater than zero.")
    
    part_len = len(content) // parts
    splits = []
    index = 0
    
    for _ in range(parts):
        splits.append(content[index: index + part_len])
        index += part_len

    return splits
