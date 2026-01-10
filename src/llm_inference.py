from utils import *

SLEEP = 5

def generate_answer_llm(image_path, prompt, tokenizer, model, max_tokens=200):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    # Prepare your prompts
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud."},
        {"role": "user", "content": prompt}
        ]   
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # generate outputs
    outputs = model.generate([text], sampling_params)[0]
    response = outputs.outputs[0].text
    return response, 0


def generate_answer_internvl3(image_path, prompt, tokenizer, model, max_tokens=200):
    generation_config = dict(max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pixel_values = load_image_internvl3(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response, 0


def generate_answer_vllm(image_paths, prompts, tokenizer, model, max_tokens=200):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens = max_tokens, 
                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    if type(image_paths)!= list or type(prompts)!= list:
        #Input is a single instance and expected output is a string
        if image_paths is None:
            inputs = [{"prompt":prompts}]
        else:
            inputs = [{"prompt":'<image>\n'+prompts, "multi_modal_data": {"image": Image.open(image_paths).convert("RGB")}}]
        outputs = model.generate(inputs, sampling_params)
        responses = outputs[0].outputs[0].text
    else:
        #Input is a batch of instance and expected output is a list
        if image_paths[0] is None:
            inputs = [{"prompt":prompts[i]} for i in range(len(prompts))]
        else:
            inputs = [{"prompt":prompts[i], "multi_modal_data": {"image": Image.open(image_paths[i]).convert("RGB")}} for i in range(len(prompts))]
        outputs = model.generate(inputs, sampling_params)
        responses = [o.outputs[0].text for o in outputs]
    return responses, 0


def generate_answer_qwen25vl(image_path, prompt, tokenizer,  model, max_tokens=200):
    if image_path:
        image = Image.open(image_path)
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, {"type": "image"}, ],},]
    else: 
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, ],},]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]   
    return response, 0


def generate_answer(image, prompt, tokenizer, model, template, max_tokens=200, use_vllm=False):
    
    prompt_map = {'internvl3':generate_answer_internvl3,
                   'qwen2.5': generate_answer_llm,
                   'qwen2.5vl': generate_answer_qwen25vl

                  }
    if use_vllm:
        answer = generate_answer_vllm(image, prompt, tokenizer, model, max_tokens)
    else:
        if type(image)==list:
            answer = []
            for im in range(len(image)):
                answer.append(prompt_map[template.split('/')[0]](image[im], prompt[im], tokenizer, model, max_tokens))
        else:
            answer = prompt_map[template.split('/')[0]](image, prompt, tokenizer, model, max_tokens)
    return answer