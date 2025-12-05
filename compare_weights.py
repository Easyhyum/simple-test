import os
import json
import torch
import numpy as np
from pathlib import Path

def load_weights(weights_file):
    """저장된 weight 파일 로드"""
    print(f"Loading weights from {weights_file}...")
    weights = torch.load(weights_file, map_location='cpu')
    return weights

def compare_weights(weights_a, weights_b, device_a_name, device_b_name):
    """두 디바이스의 weight 비교"""
    print(f"\n{'='*80}")
    print(f"Comparing weights: {device_a_name} vs {device_b_name}")
    print(f"{'='*80}\n")
    
    # 공통 레이어 찾기
    keys_a = set(weights_a.keys())
    keys_b = set(weights_b.keys())
    
    common_keys = keys_a & keys_b
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    
    print(f"Total layers in {device_a_name}: {len(keys_a)}")
    print(f"Total layers in {device_b_name}: {len(keys_b)}")
    print(f"Common layers: {len(common_keys)}")
    
    if only_in_a:
        print(f"\nLayers only in {device_a_name}: {len(only_in_a)}")
        for key in sorted(only_in_a)[:5]:
            print(f"  - {key}")
        if len(only_in_a) > 5:
            print(f"  ... and {len(only_in_a) - 5} more")
    
    if only_in_b:
        print(f"\nLayers only in {device_b_name}: {len(only_in_b)}")
        for key in sorted(only_in_b)[:5]:
            print(f"  - {key}")
        if len(only_in_b) > 5:
            print(f"  ... and {len(only_in_b) - 5} more")
    
    # 공통 레이어 비교
    comparison_results = []
    print(f"\n{'='*80}")
    print(f"Detailed Layer Comparison")
    print(f"{'='*80}\n")
    
    for key in sorted(common_keys):
        weight_a = weights_a[key]['data']
        weight_b = weights_b[key]['data']
        dtype_a = weights_a[key]['dtype']
        dtype_b = weights_b[key]['dtype']
        
        # Shape 비교
        shape_match = weight_a.shape == weight_b.shape
        
        # Dtype 비교
        dtype_match = dtype_a == dtype_b
        
        # 값 비교 (동일한 shape인 경우만)
        if shape_match:
            # 절대 차이
            abs_diff = torch.abs(weight_a - weight_b)
            max_abs_diff = float(abs_diff.max())
            mean_abs_diff = float(abs_diff.mean())
            
            # 상대 차이 (0으로 나누기 방지)
            rel_diff = abs_diff / (torch.abs(weight_a) + 1e-8)
            max_rel_diff = float(rel_diff.max())
            mean_rel_diff = float(rel_diff.mean())
            
            # 동일한 값인지 체크
            identical = torch.allclose(weight_a, weight_b, rtol=1e-5, atol=1e-8)
            
            result = {
                'layer': key,
                'shape': list(weight_a.shape),
                'dtype_a': dtype_a,
                'dtype_b': dtype_b,
                'shape_match': shape_match,
                'dtype_match': dtype_match,
                'identical': identical,
                'max_abs_diff': max_abs_diff,
                'mean_abs_diff': mean_abs_diff,
                'max_rel_diff': max_rel_diff,
                'mean_rel_diff': mean_rel_diff,
                'weight_a_mean': float(weight_a.mean()),
                'weight_b_mean': float(weight_b.mean()),
                'weight_a_std': float(weight_a.std()),
                'weight_b_std': float(weight_b.std()),
            }
        else:
            result = {
                'layer': key,
                'shape_a': list(weight_a.shape),
                'shape_b': list(weight_b.shape),
                'dtype_a': dtype_a,
                'dtype_b': dtype_b,
                'shape_match': False,
                'dtype_match': dtype_match,
                'identical': False,
            }
        
        comparison_results.append(result)
    
    # 결과 출력
    mismatched_layers = []
    for result in comparison_results:
        if not result.get('identical', False) or not result.get('dtype_match', True):
            mismatched_layers.append(result)
    
    if mismatched_layers:
        print(f"Found {len(mismatched_layers)} mismatched layers:\n")
        for result in mismatched_layers[:10]:  # 처음 10개만 자세히 출력
            print(f"Layer: {result['layer']}")
            print(f"  Shape match: {result.get('shape_match', 'N/A')}")
            print(f"  Dtype: {result['dtype_a']} vs {result['dtype_b']}")
            print(f"  Dtype match: {result.get('dtype_match', 'N/A')}")
            
            if 'max_abs_diff' in result:
                print(f"  Max abs diff: {result['max_abs_diff']:.6e}")
                print(f"  Mean abs diff: {result['mean_abs_diff']:.6e}")
                print(f"  Max rel diff: {result['max_rel_diff']:.6e}")
                print(f"  Mean rel diff: {result['mean_rel_diff']:.6e}")
                print(f"  Mean values: {result['weight_a_mean']:.6e} vs {result['weight_b_mean']:.6e}")
            print()
        
        if len(mismatched_layers) > 10:
            print(f"... and {len(mismatched_layers) - 10} more mismatched layers")
    else:
        print("✓ All layers are identical!")
    
    # 통계 요약
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}\n")
    
    shape_matched = sum(1 for r in comparison_results if r.get('shape_match', False))
    dtype_matched = sum(1 for r in comparison_results if r.get('dtype_match', True))
    identical_count = sum(1 for r in comparison_results if r.get('identical', False))
    
    print(f"Shape matched: {shape_matched}/{len(comparison_results)} ({100*shape_matched/len(comparison_results):.1f}%)")
    print(f"Dtype matched: {dtype_matched}/{len(comparison_results)} ({100*dtype_matched/len(comparison_results):.1f}%)")
    print(f"Identical weights: {identical_count}/{len(comparison_results)} ({100*identical_count/len(comparison_results):.1f}%)")
    
    if shape_matched > 0:
        valid_results = [r for r in comparison_results if 'max_abs_diff' in r]
        if valid_results:
            max_diffs = [r['max_abs_diff'] for r in valid_results]
            mean_diffs = [r['mean_abs_diff'] for r in valid_results]
            
            print(f"\nAbsolute differences across all layers:")
            print(f"  Max: {max(max_diffs):.6e}")
            print(f"  Mean: {np.mean(mean_diffs):.6e}")
            print(f"  Median: {np.median(mean_diffs):.6e}")
    
    return comparison_results

def save_comparison_report(comparison_results, output_file):
    """비교 결과를 JSON 파일로 저장"""
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nComparison report saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_weights.py <output_dir_1> <output_dir_2>")
        print("Example: python compare_weights.py ./output_results/20251205-120000 ./output_results/20251205-130000")
        sys.exit(1)
    
    dir_a = sys.argv[1]
    dir_b = sys.argv[2]
    
    # weights 디렉토리 찾기
    weights_dirs_a = [d for d in Path(dir_a).iterdir() if d.is_dir() and d.name.startswith('weights_')]
    weights_dirs_b = [d for d in Path(dir_b).iterdir() if d.is_dir() and d.name.startswith('weights_')]
    
    if not weights_dirs_a:
        print(f"No weights directory found in {dir_a}")
        sys.exit(1)
    if not weights_dirs_b:
        print(f"No weights directory found in {dir_b}")
        sys.exit(1)
    
    weights_dir_a = weights_dirs_a[0]
    weights_dir_b = weights_dirs_b[0]
    
    device_a_name = weights_dir_a.name.replace('weights_', '')
    device_b_name = weights_dir_b.name.replace('weights_', '')
    
    # 각 디렉토리의 weight 파일 찾기
    weight_files_a = list(weights_dir_a.glob('*_weights.pt'))
    weight_files_b = list(weights_dir_b.glob('*_weights.pt'))
    
    if not weight_files_a:
        print(f"No weight files found in {weights_dir_a}")
        sys.exit(1)
    if not weight_files_b:
        print(f"No weight files found in {weights_dir_b}")
        sys.exit(1)
    
    # 모델별로 비교
    for weight_file_a in weight_files_a:
        model_name = weight_file_a.stem.replace('_weights', '')
        
        # 같은 모델 이름의 파일 찾기
        weight_file_b = weights_dir_b / f"{model_name}_weights.pt"
        
        if not weight_file_b.exists():
            print(f"Warning: No matching weight file found for {model_name} in {device_b_name}")
            continue
        
        print(f"\n{'#'*80}")
        print(f"# Comparing model: {model_name}")
        print(f"{'#'*80}")
        
        weights_a = load_weights(weight_file_a)
        weights_b = load_weights(weight_file_b)
        
        comparison_results = compare_weights(weights_a, weights_b, device_a_name, device_b_name)
        
        # 비교 결과 저장
        output_file = Path(dir_a).parent / f"weight_comparison_{device_a_name}_vs_{device_b_name}_{model_name}.json"
        save_comparison_report(comparison_results, output_file)
