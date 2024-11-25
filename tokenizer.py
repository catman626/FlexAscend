import sentencepiece as spm

__all__ = ['LlamaTokenizer']



class LlamaTokenizer:
    vocabFileNames = [ 'tokenizer.model']
    
    
    def __init__(self,
        vocabFile,
        unkToken="<unk>",
        bosToken="<a>",
        eosToken="</s>",
        padToken="<unk>",
        addBosToken=True,
        addEosToken=False,
        cleanUpTokenizationSpaces=False
    ):     
        pass        

    
from mindformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("llama2_7b")
tokens = tokenizer("hello world")
print(tokens)
