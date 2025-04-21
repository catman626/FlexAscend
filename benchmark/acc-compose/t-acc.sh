### benchmark about compress
echo "" > acc

python opt.py --tokenizer model/opt-1.3b --model opt-1.3b --ckpt model/opt-1.3b/torch_model.bin --batch-size 64 --logfile acc --input-file tmp-input --response-file tmp-response


