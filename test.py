import os
import json, time, csv, gc, subprocess
# from optimum.rbln import RBLNQwen3ForCausalLM, RBLNLlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, disable_caching
    
import torch

disable_caching()
model_save_path = "/models_mlsys"
output_path = "./output_results"
if not os.path.exists(output_path):
    os.makedirs(output_path)
start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=True
    )
    device_name = result.stdout.strip().split('\n')[0]
    device_type = "cuda"
elif hasattr(torch, 'npu') and torch.npu.is_available():
    device = torch.device("npu")
    device_name = "ATOM NPU"
    device_type = "npu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple MPS"
    device_type = "mps"
else:
    device = torch.device("cpu")
    device_name = "CPU"
    device_type = "cpu"

print(f"Using device: {device_name} ({device_type})")

batch_size = 1

print("Loading configuration...")
with open("paremeter.json", "r") as f:
    config = json.load(f)

def _load_dataset():
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dir', './data')
    dataset_name = dataset_config.get('name', 'HuggingFaceH4/ultrachat_200k')

    print(f"Loading dataset from {dataset_dir}...")
    if os.path.exists(dataset_dir):
        if os.path.isdir(dataset_dir):
            print(f"Loading dataset from disk at {dataset_dir}...")
            dataset = load_from_disk(dataset_dir)
        elif dataset_dir.endswith('.jsonl') or dataset_dir.endswith('.json'):
            print(f"Loading JSONL file from {dataset_dir}...")
            dataset = load_dataset('json', data_files=dataset_dir, split='train')
        else:
            raise ValueError(f"Unsupported file format: {dataset_dir}")
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    inputs_from_config = config.get('inputs', [])
    inputs_dataset = Dataset.from_dict({"prompt_text": inputs_from_config})

    dataset = concatenate_datasets([inputs_dataset, dataset])

    return dataset

# def model_load(model_name):
#     save_path = os.path.join(model_save_path, os.path.basename(model_name))
#     if 'Qwen3' in model_name:
#         model = RBLNQwen3ForCausalLM.from_pretrained(
#             model_id=save_path,
#             export=False,
#         )
#     elif 'Llama' in model_name:
#         model = RBLNLlamaForCausalLM.from_pretrained(
#             model_id=save_path,
#             export=False,
#         )
#     else:
#         raise ValueError(f"Unsupported model name: {model_name}")
#     return model

def model_load(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model

def save_csv(file_path, data, headers):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

if __name__ == "__main__":
    # Load dataset configuration from hyperparameter.json
    dataset = _load_dataset()
    print(f"Dataset length: {len(dataset)}")

    dataset_length = config.get('request_number', len(dataset))
    final_dataset = dataset.select(range(dataset_length))

    model_list = config.get('models', [])
    print(model_list)

    max_new_tokens = config.get('max_new_tokens', 40960)

    csv_headers = ['device', 'model', 'type', 'batch_size', 'data_index', 'input_text', 'input_tokens', 'output_text', 'output_tokens']
    csv_file_path = os.path.join(output_path, f"{device_name.replace(' ', '_')}_input_output_summary_{start_time}.csv")

    for model_name in model_list:
        print(f"Loading model: {model_name}...")
        model = model_load(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        for idx, input_ids in enumerate(final_dataset):
            prompt_text = input_ids['prompt_text']
            print(f"<<{idx}. input: {prompt_text},")
            input_ids = tokenizer(prompt_text, return_tensors="pt")
            
            # 입력 토큰화
            input_token_ids = input_ids['input_ids'][0].tolist()
            input_tokens_text = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_token_ids]
            input_ids = {key: value.to(device) for key, value in input_ids.items()}
            # 텍스트 전처리: 콤마를 <comma>로, 줄바꿈을 <br>로 변경
            processed_input_text = "|||".join([t.replace(',', ' <comma> ').replace('\n', ' <br> ') for t in input_tokens_text])
            processed_input_tokens = "|||".join([str(t) for t in input_token_ids])
            
            # print(f"{idx}. Attention mask shape:{input_ids['attention_mask'].shape}, {input_ids['attention_mask']}")
            print(f"<<{idx}. Generated:", end='', flush=True)
            # current_attention_mask = input_ids['attention_mask']
            outputs = model.generate(
                input_ids=input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],  # 중요: attention_mask 명시적으로 전달
                max_new_tokens=max_new_tokens,
                do_sample=False,  
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
            # 출력 토큰화 (입력 부분 제외)
            output_token_ids = outputs[0][input_ids['input_ids'].shape[-1]:].tolist()
            output_tokens_text = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in output_token_ids]
            
            # 출력 텍스트 전처리
            processed_output_text = "|||".join([t.replace(',', ' <comma> ').replace('\n', ' <br> ') for t in output_tokens_text])
            processed_output_tokens = "|||".join([str(t) for t in output_token_ids])
            
            generated_texts = tokenizer.decode(
                outputs[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(generated_texts)
            
            # CSV 저장
            csv_data = {
                'device': device_name,
                'model': model_name,
                'type': device_type,
                'batch_size': batch_size,
                'data_index': idx,
                'input_text': processed_input_text,
                'input_tokens': processed_input_tokens,
                'output_text': processed_output_text,
                'output_tokens': processed_output_tokens
            }
            save_csv(csv_file_path, csv_data, csv_headers)
            
            print('-' * 80)
            print()
        
        # 모델 메모리 해제
        del model
        del tokenizer
        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        print(f"Model {model_name} unloaded and memory cleared.")
        
            # past_kv = None
            # for decoding_idx in range(max_new_tokens): 
            #     outputs = model(
            #         input_ids=input_ids['input_ids'], 
            #         attention_mask=current_attention_mask,
            #         past_key_values = past_kv if decoding_idx > 0 else None,
            #         use_cache=True,
            #         return_dict=True
            #     )
            #     print(outputs)
            #     next_token_ids = torch.argmax(outputs.logits, dim=-1, keepdim=True)[0][0][0]
            #     text_token = tokenizer.decode(next_token_ids, skip_special_tokens=True)
            #     past_kv = outputs.past_key_values

            #     input_ids = tokenizer(text_token, return_tensors="pt")
            #     if '\n' in text_token:
            #         display_text = text_token.replace('\n', ' \\n ')
            #     else:
            #         display_text = text_token

            #     print(display_text, end='', flush=True)

            #     current_attention_mask = torch.cat(
            #         [current_attention_mask, torch.ones((1,1), dtype=current_attention_mask.dtype)], dim=-1
            #     )
            #     print(f"{idx}.{decoding_idx} Current attention mask shape:{current_attention_mask.shape}, {current_attention_mask}")

            # print('-' * 80)