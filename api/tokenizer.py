from transfromers import AutoTokenizer



tokenizer = AutoTokenizer("model/opt-1.3b")
inputs = [
    "hello world!",
    "the largest cat in the world is "
]
inputTokens = tokenizer(inputs, padding="max_length",  max_length=2048).input_ids
print(inputTokens)