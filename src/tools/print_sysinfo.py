import platform
import os
import datetime
import pynvml
import psutil




def print_env():
    print('========== System Information ==========')
    # 오늘의 날짜
    current_date = datetime.date.today()
    print(f'DATE : {current_date}')
    
    # Python 버전
    python_version = platform.python_version()
    print(f'Pyton Version : {python_version}')
    
    # Pytorch 버전
    # 이 환경에 PyTorch가 설치되어 있는지 확인합니다.
    try:
        import torch
        pytorch_version = torch.__version__
        
    except ImportError:
        pytorch_version = "PyTorch not installed"

    print(f'PyTorch Version : {pytorch_version}')
    # 현재 작업환경의 os
    os_info = platform.system() + " " + platform.release()
    print(f'OS : {os_info}')
        
    # 현재 작업환경의 CPU 스펙
    cpu_info = platform.processor()
    print(f'CPU spec : {cpu_info}')
    
    # 현재 작업환경의 Memory 스펙
    mem_info = psutil.virtual_memory().total
    print(f'RAM spec : {mem_info / (1024**3):.2f} GB')
    
    # 현재 작업환경의 GPU 스펙
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        
        print(f"Device {i}:")
        print(f"Name: {name}")
        print(f"Total Memory: {memory_info.total / 1024**2} MB")
        print(f"Driver Version: {driver_version}")
        print("="*30)

    pynvml.nvmlShutdown()
