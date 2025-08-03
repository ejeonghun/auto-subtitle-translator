#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
디바이스 설정 유틸리티
CUDA → MPS → CPU 순서로 최적의 디바이스 자동 선택
"""

import os
import platform
from typing import Tuple, List, Dict, Any

def get_optimal_device() -> Tuple[str, Dict[str, Any]]:
    """
    최적의 디바이스를 자동으로 선택
    우선순위: CUDA → MPS (Apple Silicon) → CPU
    
    Returns:
        (device_name, device_info) 튜플
    """
    device_info = {
        "type": "cpu",
        "name": "CPU",
        "memory": None,
        "compute_capability": None,
        "torch_available": False,
        "onnx_providers": []
    }
    
    try:
        import torch
        device_info["torch_available"] = True
        
        # 1. CUDA 확인 (NVIDIA GPU)
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown CUDA GPU"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory if gpu_count > 0 else 0
            compute_capability = torch.cuda.get_device_capability(0) if gpu_count > 0 else None
            
            device_info.update({
                "type": "cuda",
                "name": gpu_name,
                "memory": gpu_memory,
                "compute_capability": compute_capability,
                "device_count": gpu_count
            })
            
            print(f"🚀 CUDA 디바이스 감지 - GPU: {gpu_name}")
            print(f"   메모리: {gpu_memory / (1024**3):.1f}GB, 디바이스 수: {gpu_count}")
            return "cuda", device_info
        
        # 2. MPS 확인 (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info.update({
                "type": "mps",
                "name": "Apple Silicon GPU (MPS)"
            })
            
            print("🍎 Apple Silicon GPU (MPS) 디바이스 감지")
            return "mps", device_info
            
    except ImportError:
        print("⚠️ PyTorch가 설치되지 않아 GPU 확인을 건너뜁니다.")
    
    # 3. CPU 백업
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    device_info.update({
        "type": "cpu",
        "name": f"CPU ({cpu_count} cores)",
        "cpu_count": cpu_count
    })
    
    print(f"💻 CPU 디바이스 사용 - {cpu_count}개 코어")
    return "cpu", device_info

def get_onnx_providers(device_type: str = None) -> List[Any]:
    """
    ONNX Runtime 실행 제공자를 디바이스에 맞게 설정
    
    Args:
        device_type: 강제 디바이스 타입 (None이면 자동 감지)
        
    Returns:
        ONNX Runtime 제공자 리스트
    """
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        providers = []
        
        if device_type is None:
            device_type, _ = get_optimal_device()
        
        # CUDA 제공자 설정
        if device_type == "cuda" and 'CUDAExecutionProvider' in available_providers:
            try:
                import torch
                if torch.cuda.is_available():
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB 제한
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
                    print("🚀 ONNX Runtime CUDA 제공자 설정")
            except ImportError:
                pass
        
        # DirectML 제공자 설정 (Windows GPU 백업)
        elif 'DmlExecutionProvider' in available_providers and platform.system() == "Windows":
            providers.append('DmlExecutionProvider')
            print("🔷 ONNX Runtime DirectML 제공자 설정 (Windows GPU)")
        
        # CoreML 제공자 설정 (macOS)
        elif 'CoreMLExecutionProvider' in available_providers and platform.system() == "Darwin":
            providers.append('CoreMLExecutionProvider')
            print("🍎 ONNX Runtime CoreML 제공자 설정 (macOS)")
        
        # CPU 백업 제공자
        providers.append('CPUExecutionProvider')
        
        if len(providers) == 1 and providers[0] == 'CPUExecutionProvider':
            print("💻 ONNX Runtime CPU 제공자 사용")
        
        return providers
        
    except ImportError:
        print("⚠️ onnxruntime가 설치되지 않았습니다.")
        return ['CPUExecutionProvider']

def get_whisper_device_config(device_type: str = None) -> Dict[str, Any]:
    """
    Faster Whisper 디바이스 설정
    
    Args:
        device_type: 강제 디바이스 타입 (None이면 자동 감지)
        
    Returns:
        Whisper 설정 딕셔너리
    """
    if device_type is None:
        device_type, device_info = get_optimal_device()
    
    config = {
        "device": "cpu",
        "compute_type": "int8"
    }
    
    if device_type == "cuda":
        # CUDA 디바이스에서 float16 지원 여부 확인
        try:
            import torch
            if torch.cuda.is_available():
                # GPU 아키텍처 확인
                device_props = torch.cuda.get_device_properties(0)
                compute_capability = torch.cuda.get_device_capability(0)
                
                # Compute Capability 7.0 이상에서 효율적인 float16 지원
                if compute_capability[0] >= 7:
                    config.update({
                        "device": "cuda",
                        "compute_type": "float16"
                    })
                    print(f"🚀 Faster Whisper CUDA 설정 (float16 지원, CC: {compute_capability[0]}.{compute_capability[1]})")
                else:
                    # 구형 GPU는 int8 사용
                    config.update({
                        "device": "cuda", 
                        "compute_type": "int8"
                    })
                    print(f"🚀 Faster Whisper CUDA 설정 (int8 사용, CC: {compute_capability[0]}.{compute_capability[1]})")
            else:
                print("⚠️ CUDA 사용 불가, CPU로 대체")
        except Exception as e:
            print(f"⚠️ CUDA 설정 중 오류 발생, CPU로 대체: {e}")
            
    elif device_type == "mps":
        # MPS는 직접 지원하지 않으므로 CPU 사용
        config.update({
            "device": "cpu",
            "compute_type": "int8"
        })
        print("🍎 Faster Whisper CPU 설정 (MPS 미지원)")
    else:
        print("💻 Faster Whisper CPU 설정")
    
    return config

def get_torch_device(device_type: str = None) -> str:
    """
    PyTorch 디바이스 문자열 반환
    
    Args:
        device_type: 강제 디바이스 타입 (None이면 자동 감지)
        
    Returns:
        PyTorch 디바이스 문자열
    """
    if device_type is None:
        device_type, _ = get_optimal_device()
    
    if device_type == "cuda":
        return "cuda"
    elif device_type == "mps":
        return "mps"
    else:
        return "cpu"

def print_device_summary():
    """디바이스 정보 요약 출력"""
    device_type, device_info = get_optimal_device()
    
    print("\n" + "="*50)
    print("🖥️  디바이스 정보 요약")
    print("="*50)
    print(f"선택된 디바이스: {device_info['name']}")
    print(f"타입: {device_type.upper()}")
    
    if device_info.get("memory"):
        print(f"메모리: {device_info['memory'] / (1024**3):.1f}GB")
    
    if device_info.get("compute_capability"):
        major, minor = device_info['compute_capability']
        print(f"Compute Capability: {major}.{minor}")
    
    if device_info.get("cpu_count"):
        print(f"CPU 코어 수: {device_info['cpu_count']}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    print_device_summary()