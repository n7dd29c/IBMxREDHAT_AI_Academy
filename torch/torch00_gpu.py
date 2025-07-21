import torch

print('pytorch version :', torch.__version__)

cuda_available = torch.cuda.is_available()
print('CUDA 사용 여부 :', cuda_available)

gpu_count = torch.cuda.device_count()
print('사용 가능한 GPU 수 :', gpu_count)

if cuda_available:
    current_device = torch.cuda.current_device()
    print('현재 사용중인 GPU 장치 :', current_device)
    print('현재 GPU 이름 :', torch.cuda.get_device_name(current_device))
else:
    print('GPU 없음')
    
print('CUDA version :', torch.version.cuda)

cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None:
    print('cuDNN version', cudnn_version)
else:
    print('cuDNN 없음')