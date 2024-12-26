- `api`  
  - test api in mindspore  
- `model` 
  - only configurations  
  - no weight files, exculude in .gitignore  
    - weight sync via obs, 
    - zip -> obs -> modelarts -> unzip  
  - structure  
    model   
    ├── convert_weight.py    
    ├── opt-1.3b.zip  
    └── opt-125m  
        ├── config.json  
        ├── generation_config.json  
        ├── merges.txt  
        ├── mindspore_model.ckpt  
        ├── pytorch_model.bin  
        ├── special_tokens_map.json  
        ├── tokenizer_config.json  
        └── vocab.json    
    
    2 directories, 22 files  
