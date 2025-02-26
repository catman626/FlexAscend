# python opt.py --ckpt model/opt-125m/mindspore_model.ckpt --tokenizer model/opt-125m --model opt-125m 
python opt.py --ckpt model/opt-125m/mindspore_model.ckpt --tokenizer model/opt-125m --model opt-125m --offload --prefetch


# multiple checkpoint files
# python opt.py --ckpt model/opt-6.7b/mindspore_model.0.ckpt model/opt-6.7b/mindspore_model.1.ckpt --tokenizer model/opt-1.3b --model opt-6.7b --offload
# should throw exception
# python opt.py --ckpt model/opt-6.7b/mindspore_model.0.ckpt model/opt-6.7b/mindspore_model.1.ckpt --tokenizer model/opt-1.3b --model opt-6.7b --offload



# python opt.py --tokenizer model/opt-1.3b --model opt-1.3b
# python opt.py --tokenizer model/opt-1.3b --model opt-175b --offload
