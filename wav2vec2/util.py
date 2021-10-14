from hangul_utils import join_jamos


def remove_duplicate_tokens(token_list):
    prev_token = -1
    clean_token_list = []
    for token in token_list:
        if token == 49:
            prev_token = -1
        elif token != prev_token:
            prev_token = token
            clean_token_list.append(token)

    return clean_token_list


def decode_CTC(token_list, tokenizer):
    clean_token_list = remove_duplicate_tokens(token_list)
    raw_char_list = list(map(tokenizer.convert, clean_token_list))
    joined_string = join_jamos(''.join(raw_char_list))
    return list(joined_string)
