from transformers import AutoTokenizer

tokenizer = AutoTokenizer("model/opt-30b")
inputs = [
    "hello world!",
    "the largest cat in the world is "
]
# inputTokens = tokenizer(inputs, padding="max_length",  max_length=2048).input_ids
inputTokens = tokenizer(inputs).input_ids

print(inputTokens)