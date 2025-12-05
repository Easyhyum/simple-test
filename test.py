import os
import json, time, csv, gc, subprocess

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, disable_caching
    
import torch

disable_caching()
model_save_path = "/models_mlsys"
output_path = "./output_results"
if not os.path.exists(output_path):
    os.makedirs(output_path)
start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

output_path = f"./output_results/{start_time}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

batch_size = 1
tp_num = 1
device = None
device_name = None

try:
    output = subprocess.run(['rbln-stat'], capture_output=True, text=True, check=True)
    if 'RBLN-CA22'  in output.stdout:
        device_name = "ATOM NPU"
        device_type = "npu"
except:
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
with open("parameter.json", "r") as f:
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

def model_load_npu(model_name):
    from optimum.rbln import RBLNQwen3ForCausalLM, RBLNLlamaForCausalLM
    save_path = os.path.join(model_save_path, f"batch_size{batch_size}_tp{tp_num}")
    print(f"model_path: {save_path}")
    save_path = os.path.join(save_path, os.path.basename(model_name))
    print(f"model_path: {save_path}")
    if 'Qwen3' in model_name:
        model = RBLNQwen3ForCausalLM.from_pretrained(
            model_id=save_path,
            export=False,
        )
    elif 'Llama' in model_name:
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=save_path,
            export=False,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

def model_load(model_name):
    qantized_flag = True if 'FP8' in model_name or 'quantized' in model_name or '4bit' in model_name or 'w8a8' in model_name else False
    if qantized_flag:
        if 'w8a8' in model_name:
            from transformers import modeling_utils
            original_load_param = modeling_utils._load_parameter_into_model
            
            def patched_load_param(model, param_name, tensor):
                # For int8 tensors or any non-float tensors, set requires_grad=False explicitly
                if tensor.dtype in [torch.int8, torch.uint8] or not tensor.is_floating_point():
                    module_name, param_type = param_name.rsplit(".", 1)
                    module = model
                    for name in module_name.split("."):
                        module = getattr(module, name)
                    # Directly set the parameter/buffer without gradient checking
                    param = torch.nn.Parameter(tensor, requires_grad=False)
                    if param_type == "weight":
                        module.weight = param
                    elif param_type == "bias":
                        module.bias = param
                    else:
                        # For scale, zero_point, etc., set as attribute
                        setattr(module, param_type, param)
                else:
                    original_load_param(model, param_name, tensor)
            
            modeling_utils._load_parameter_into_model = patched_load_param
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    device_map=None,  # CPU에 먼저 로드
                    low_cpu_mem_usage=True,
                )
                
                # After loading, replace forward method for INT8 quantized layers
                quantized_layer_count = 0
                
                def create_dequant_forward(original_forward, weight_scale):
                    def dequant_forward(self, input):
                        # Dequantize on-the-fly
                        x = input
                        target_dtype = x.dtype
                        
                        # Dequantize weight
                        weight_float = self.weight.to(target_dtype) * weight_scale.to(target_dtype)
                        
                        # Perform linear operation
                        return torch.nn.functional.linear(x, weight_float, self.bias)
                    return dequant_forward
                
                # Replace forward method for all quantized Linear layers
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_scale') and module.weight.dtype == torch.int8:
                        # Save weight_scale
                        weight_scale = module.weight_scale
                        # Replace the forward method
                        module.forward = create_dequant_forward(module.forward, weight_scale).__get__(module, type(module))
                        quantized_layer_count += 1
                
                print(f"Replaced forward method for {quantized_layer_count} quantized Linear layers")
                        
            finally:
                modeling_utils._load_parameter_into_model = original_load_param
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # FP16으로 로드
        )
        model = model.to(torch.float16)
    model.eval()
    model.to(device)
    print(f"Model dtype: {next(model.parameters()).dtype}")
    return model

def save_csv(file_path, data, headers):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def save_model_weights(model, model_name, device_type, output_dir):
    """모델의 모든 레이어 weight를 저장"""
    weights_dir = os.path.join(output_dir, f"weights_{device_type}")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 전체 모델 이름 유지 (/ -> _로 변환)
    model_basename = model_name.replace('/', '_')
    weights_file = os.path.join(weights_dir, f"{model_basename}_weights.pt")

    weight_model = model
    
    # 모든 파라미터를 CPU로 이동하여 저장
    weights_dict = {}
    try:
        for name, param in weight_model.named_parameters():
            weights_dict[name] = {
                'data': param.detach().cpu().clone(),
                'shape': list(param.shape),
                'dtype': str(param.dtype)
            }
    except Exception as e:
        print(f"Error extracting weights: {e}")
        if device_type == 'npu':
            del hf_model
        return None
    
    torch.save(weights_dict, weights_file)
    print(f"Saved {len(weights_dict)} layer weights to {weights_file}")
    
    # 통계 정보 저장
    stats_file = os.path.join(weights_dir, f"{model_basename}_stats.json")
    stats = {}
    for name, info in weights_dict.items():
        tensor = info['data']
        stats[name] = {
            'shape': info['shape'],
            'dtype': info['dtype'],
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'num_params': int(tensor.numel())
        }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved weight statistics to {stats_file}")
    
    # NPU용 HuggingFace 모델 메모리 해제
    if device_type == 'npu':
        del hf_model
        gc.collect()
    
    return weights_file

def load_kv_cache(model_name, input_index, load_from_device, device):
    """저장된 KV cache 로드"""
    model_basename = model_name.replace('/', '_')
    model_dir = os.path.join(load_from_device, model_basename)
    kv_file = os.path.join(model_dir, f"input{input_index}.pt")
    
    if not os.path.exists(kv_file):
        print(f"Warning: KV cache file not found: {kv_file}")
        raise FileNotFoundError(f"KV cache file not found: {kv_file}")
    
    print(f"Loading KV cache from {kv_file}")
    kv_data = torch.load(kv_file, map_location='cpu')
    
    # KV cache를 DynamicCache 객체로 변환
    if kv_data['kv_cache'] is not None:
        cache = DynamicCache()
        # 레이어별로 key, value 추가
        for layer_idx, layer_data in enumerate(kv_data['kv_cache']):
            key = layer_data['key'].to(device) if device else layer_data['key']
            value = layer_data['value'].to(device) if device else layer_data['value']
            # update 메서드 호출 (layer_idx 명시)
            cache.update(key, value, layer_idx)
        kv_cache = cache
    else:
        kv_cache = None
    
    # input_tokens는 이미 list 형태
    input_token_ids = kv_data['input_tokens']
    output_token_idx = kv_data['output_tokens']
    output_token_text = kv_data['output_text']
    return kv_cache, input_token_ids, kv_data, output_token_idx, output_token_text

def save_kv_cache(kv_cache, input_text, input_tokens, output_text, output_tokens, 
                  model_name, device_name, input_index, output_dir):
    """KV cache와 입출력 데이터를 저장"""
    # 디바이스별 디렉토리 생성
    device_dir = os.path.join(output_dir, device_name.replace(' ', '_'))
    
    # 모델별 폴더 생성 (/ -> _로 변환)
    model_basename = model_name.replace('/', '_')
    model_dir = os.path.join(device_dir, model_basename)
    os.makedirs(model_dir, exist_ok=True)
    
    # KV cache 파일명
    kv_file = os.path.join(model_dir, f"input{input_index}.pt")
    
    # KV cache 데이터 준비
    kv_data = {
        'input_text': input_text,
        'input_tokens': input_tokens,
        'output_text': output_text,
        'output_tokens': output_tokens,
        'model_name': model_name,
        'device': device_name,
        'input_index': input_index,
    }
    
    # KV cache 저장 (CPU로 이동)
    if kv_cache is not None:
        kv_list = []
        # DynamicCache를 legacy tuple 형식으로 변환
        if hasattr(kv_cache, 'to_legacy_cache'):
            legacy_cache = kv_cache.to_legacy_cache()
        else:
            # DynamicCache를 직접 접근하여 tuple로 변환
            try:
                legacy_cache = [(kv_cache[i][0], kv_cache[i][1]) for i in range(len(kv_cache))]
            except:
                # 다른 방법으로 접근
                legacy_cache = []
                for i in range(len(kv_cache)):
                    legacy_cache.append((kv_cache[i][0], kv_cache[i][1]))
        
        for layer_idx, (key, value) in enumerate(legacy_cache):
            key_tensor = key.detach().cpu().clone()
            value_tensor = value.detach().cpu().clone()
            kv_list.append({
                'key': key_tensor,
                'value': value_tensor,
                'key_shape': list(key_tensor.shape),
                'value_shape': list(value_tensor.shape),
                'key_dtype': str(key_tensor.dtype),
                'value_dtype': str(value_tensor.dtype),
            })
        kv_data['kv_cache'] = kv_list
        kv_data['num_layers'] = len(kv_list)
    else:
        kv_data['kv_cache'] = None
        kv_data['num_layers'] = 0
    
    # 파일 저장
    torch.save(kv_data, kv_file)
    print(f"Saved KV cache to {kv_file}")
    
    # 메타데이터 JSON 저장 (KV cache 텐서 제외)
    meta_file = os.path.join(model_dir, f"input{input_index}_meta.json")
    meta_data = {
        'input_text': input_text,
        'input_tokens': input_tokens,
        'output_text': output_text,
        'output_tokens': output_tokens,
        'model_name': model_name,
        'device': device_name,
        'input_index': input_index,
        'num_layers': kv_data['num_layers'],
    }
    
    if kv_cache is not None and len(kv_data['kv_cache']) > 0:
        meta_data['kv_shapes'] = [
            {
                'layer': i,
                'key_shape': item['key_shape'],
                'value_shape': item['value_shape'],
                'key_dtype': item['key_dtype'],
                'value_dtype': item['value_dtype'],
            }
            for i, item in enumerate(kv_data['kv_cache'])
        ]
    
    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f"Saved metadata to {meta_file}")
    
    return kv_file

if __name__ == "__main__":
    # Load dataset configuration from hyperparameter.json
    dataset = _load_dataset()
    print(f"Dataset length: {len(dataset)}")

    dataset_length = config.get('request_number', len(dataset))
    final_dataset = dataset.select(range(dataset_length))

    model_list = config.get('models', [])
    print(model_list)

    max_new_tokens = config.get('max_new_tokens', 40960)
    
    # KV cache 설정
    kv_cache_config = config.get('kv_cache', {})
    save_kv = kv_cache_config.get('save', False)
    load_kv = kv_cache_config.get('load', False)
    load_from_device = kv_cache_config.get('load_from_device', '')
    
    print(f"KV Cache Config - Save: {save_kv}, Load: {load_kv}")
    if load_kv:
        print(f"Loading KV cache from: {load_from_device}")

    csv_headers = ['device', 'model', 'type', 'batch_size', 'data_index', 'input_text', 'input_tokens', 'output_text', 'output_tokens', 'mismatch', 'mismatch_index']
    csv_file_path = os.path.join(output_path, f"{device_name.replace(' ', '_')}_input_output_summary_{start_time}.csv")

    for model_name in model_list:
        print(f"Loading model: {model_name}...")
        if 'npu' in device_type:
            if '8B' in model_name:
                print(f"Using tp=2 for model {model_name}")
                tp_num = 2
            else:
                print(f"Using tp=1 for model {model_name}")
                tp_num = 1
            model = model_load_npu(model_name)
        else:
            model = model_load(model_name)
        print(f"Model device location: {next(model.parameters()).device}")
        
        # print(model)
        # continue
        # 모델 weight 저장
        # weights_file = save_model_weights(model, model_name, device_type, output_path)
        
        # continue
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        for idx, input_ids in enumerate(final_dataset):
            prompt_text = input_ids['prompt_text']
            
            input_ids = tokenizer(prompt_text, return_tensors="pt")
            
            # 입력 토큰화
            input_token_ids = input_ids['input_ids'][0].tolist()
            input_tokens_text = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_token_ids]
            
            # KV cache 로드 여부 확인
            past_key_values = None
            original_input_length = len(input_token_ids)
            
            if load_kv:
                loaded_kv, loaded_input_tokens, loaded_data, loaded_output_token_idx, loaded_output_token_text = load_kv_cache(
                    model_name=model_name,
                    input_index=idx,
                    load_from_device=load_from_device,
                    device=device
                )
                
                if loaded_kv is not None and loaded_input_tokens is not None:
                    # 로드된 입력과 현재 입력이 일치하는지 확인
                    if loaded_input_tokens == input_token_ids:
                        print(f"\n[KV Cache Loaded] Using cached KV for {len(loaded_input_tokens)-1} tokens")
                        print(f"Original input length: {len(input_token_ids)}, Using last token only")
                        
                        # 마지막 토큰만 사용
                        last_token_id = input_token_ids[-1]
                        input_ids['input_ids'] = torch.tensor([[last_token_id]], dtype=torch.long)
                        
                        # Attention mask: 전체 시퀀스 길이에 대해 1로 설정
                        input_ids['attention_mask'] = torch.ones((1, original_input_length), dtype=torch.long)
                        
                        past_key_values = loaded_kv
                    else:
                        print(f"\n[Warning] Input mismatch! Loaded: {len(loaded_input_tokens)} tokens, Current: {len(input_token_ids)} tokens")
                        print("Proceeding without cached KV...")
            
            # 입력을 FP16으로 변환하여 device로 이동
            if device is not None:
                input_ids = {key: value.to(device) for key, value in input_ids.items()}
            
            print(f"\n[Generating] Input shape: {input_ids['input_ids'].shape}, Attention mask shape: {input_ids['attention_mask'].shape}")
            if past_key_values is not None:
                print(f"[Using Past KV] Layers: {len(past_key_values)}, Full KV Key shape: {past_key_values[0][0].shape}")

            # 토큰별 생성 (KV cache를 매 iteration마다 사용)
            current_input_ids = input_ids['input_ids']
            current_attention_mask = input_ids['attention_mask']
            
            generated_token_ids = []
            output_tokens_text = []
            kv_cache = None
            mismatch_found = False
            mismatch_index = -1
            for step in range(max_new_tokens):
                # KV cache 슬라이싱 (필요한 길이만큼만 사용)
                if past_key_values is not None:
                    kv_length = original_input_length + step - 1
                    
                    # DynamicCache를 슬라이싱
                    # DynamicCache를 legacy tuple 형식으로 변환
                    if hasattr(past_key_values, 'to_legacy_cache'):
                        legacy_cache = past_key_values.to_legacy_cache()
                    else:
                        # DynamicCache를 직접 접근
                        legacy_cache = [(past_key_values[i][0], past_key_values[i][1]) for i in range(len(past_key_values))]
                    
                    # 슬라이싱 후 다시 DynamicCache로 변환
                    current_kv = DynamicCache()
                    for layer_idx, (key, value) in enumerate(legacy_cache):
                        sliced_key = key[:, :, :kv_length, :]
                        sliced_value = value[:, :, :kv_length, :]
                        current_kv.update(sliced_key, sliced_value, layer_idx)
                    
                    current_attention_mask = torch.ones((1, kv_length + 1), dtype=torch.long, device=device if device else current_input_ids.device)
                    
                    if step == 0:
                        print(f"\n[Step {step}] Using KV cache length: {kv_length} (sliced from {legacy_cache[0][0].shape[2]}), Attention mask length: {current_attention_mask.shape[1]}")
                else:
                    if step == 0:
                        current_kv = None
                    current_attention_mask = torch.ones((1, original_input_length + step), dtype=torch.long, device=device if device else current_input_ids.device)

                if step == 0:
                    if past_key_values is not None:
                        print(f"<<{idx}. Attention Mask: {current_attention_mask.shape}, KV used with length {kv_length}, Original input length: {original_input_length}")
                    else:
                        print(f"<<{idx}. Attention Mask: {current_attention_mask.shape}")
                    print(f"<<{idx}. input: {prompt_text}")
                    print(f"<<{idx}. Generated:", end='', flush=True)

                # forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=current_kv,
                        use_cache=True,
                        return_dict=True
                    )
                
                # greedy decoding
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 마지막에 저장된 kv_cache 업데이트
                if step == max_new_tokens - 1 and past_key_values is None:
                    kv_cache = outputs.past_key_values
                
                if not load_kv:
                    current_kv = outputs.past_key_values

                generated_token_ids.append(next_token_id.item())
                token_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
                output_tokens_text.append(token_text)
                
                if load_kv:
                    if loaded_output_token_idx[step] != next_token_id.item():
                        print(f"\n[Warning] mismatch step:{step+1} Loaded: {loaded_output_token_text[step]} ({loaded_output_token_idx[step]}), Generated: {token_text} ({next_token_id.item()})")
                        mismatch_found = True
                        mismatch_index = step
                        break

                # 진행상황 출력
                if '\n' in token_text:
                    print(' \\n ', end='', flush=True)
                else:
                    print(token_text, end='', flush=True)
                
                # EOS 토큰 체크
                eos_token_id = tokenizer.eos_token_id
                if next_token_id.item() == eos_token_id:
                    print(f"\n[EOS reached at step {step+1}]")
                    break
            
                current_input_ids = next_token_id
            
            print()
            
            generated_texts = tokenizer.decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(f"\n[Generated text]: {generated_texts}")

            # KV cache 저장 (save=True인 경우만)
            if save_kv:
                # past_key_values를 사용한 경우, 원본 입력의 KV cache를 저장해야 하므로
                # 여기서는 로드하지 않고 생성한 경우에만 저장
                if past_key_values is None and kv_cache is not None:
                    save_kv_cache(
                        kv_cache=kv_cache,
                        input_text=input_tokens_text,  # list 형태로 저장
                        input_tokens=input_token_ids,  # list 형태로 저장
                        output_text=output_tokens_text,  # list 형태로 저장
                        output_tokens=generated_token_ids,  # list 형태로 저장
                        model_name=model_name,
                        device_name=device_name,
                        input_index=idx,
                        output_dir=output_path
                    )
                else:
                    print(f"[KV Cache] Skipped saving (loaded from cache or no cache available)")
            
            # CSV 저장용 텍스트 전처리
            processed_input_text = "|||".join([t.replace(',', ' <comma> ').replace('\n', ' <br> ') for t in input_tokens_text])
            processed_input_tokens = "|||".join([str(t) for t in input_token_ids])
            processed_output_text = "|||".join([t.replace(',', ' <comma> ').replace('\n', ' <br> ') for t in output_tokens_text])
            processed_output_tokens = "|||".join([str(t) for t in generated_token_ids])
            
            # CSV 저장
            csv_data = {
                'device': f"{device_name}_Aware" if load_kv else device_name,
                'model': model_name,
                'type': device_type,
                'batch_size': batch_size,
                'data_index': idx,
                'input_text': processed_input_text,
                'input_tokens': processed_input_tokens,
                'output_text': processed_output_text,
                'output_tokens': processed_output_tokens,
                'mismatch': mismatch_found,
                'mismatch_index': mismatch_index
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
