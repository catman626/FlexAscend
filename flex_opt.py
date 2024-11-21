from transformers import AutoTokenizer
class FlexOpt:
    def __init__(self, config, ):
        pass


def testInputs(promptLength, numPrompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    inputIdces = tokenizer(prompts, padding="max_length", max_length=promptLength)
    
    return (inputIdces[0], ) * numPrompts

def runFlexOpt():
    print("run flex OPT model")

    tokenizer = AutoTokenizer.from_pretrained("~/inference/model/opt-1.3b")

    
