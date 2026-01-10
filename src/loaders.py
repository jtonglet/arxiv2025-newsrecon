from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoProcessor
import torch


def loader_llm_default(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, 
                                                 torch_dtype=torch.float16)
    tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def loader_vllm(model_path):
    from vllm import LLM
    model = LLM(model_path,
                tokenizer= model_path,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.7,
                trust_remote_code=True,
                max_model_len=10000)
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def loader_internvl3(model_path):
    torch.cuda.empty_cache()
    if '38B' not in model_path and '78B' not in model_path:
        model = AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16, 
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto").eval()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def loader_qwen25vl(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration

    if '32B' not in model_path and '72B' not in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
        )   
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  
            llm_int8_enable_fp32_cpu_offload=True  
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto"
            )  

    tokenizer = AutoProcessor.from_pretrained(model_path)
    return model, tokenizer


def load_model(model, use_vllm=False):
    paths = {
    'internvl3/1B/':'models--opengvlab--InternVL3/1B/',
    'internvl3/2B/':'models--opengvlab--InternVL3/2B/',
    'internvl3/8B/':'models--opengvlab--InternVL3/8B/',
    'internvl3/14B/':'models--opengvlab--InternVL3/14B/',
    'internvl3/38B/':'models--opengvlab--InternVL3/38B/',
    'internvl3/78B/':'models--opengvlab--InternVL3/78B/',
    'qwen2.5/7B/': 'models--Qwen2.5-7B-Instruct/' ,
    'qwen2.5vl/3B/': 'Qwen2.5-VL-3B-Instruct/',
    'qwen2.5vl/7B/': 'Qwen2.5-VL-7B-Instruct/',
    'qwen2.5vl/32B/': 'Qwen2.5-VL-32B-Instruct/',
    'qwen2.5vl/72B/': 'Qwen2.5-VL-72B-Instruct/',
        }
    
    loader_map = {'internvl3': loader_internvl3, 'qwen2.5': loader_llm_default, 'qwen2.5vl':loader_qwen25vl}
    if use_vllm:
        model, tokenizer = loader_vllm(paths[model])
    else:
        model, tokenizer = loader_map[model.split('/')[0]](paths[model])
    return model, tokenizer