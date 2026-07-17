"""Code for tokenising TinyStories for training."""

from transformers import AutoTokenizer

tokeniser = AutoTokenizer.from_pretrained("QWEN/Qwen3-0.6B")

print(tokeniser.model_max_length)
print(tokeniser.vocab_size)
print(tokeniser.special_tokens_map)

text = "Hello, my name is Cameron, this is a test."

tokens = tokeniser.tokenize(text)
ids = tokeniser.convert_tokens_to_ids(tokens)

print(tokens)
print(ids)