import torch
print("1. 显卡驱动状态:", torch.cuda.is_available())
print("2. 识别到的显卡:", torch.cuda.get_device_name(0))
print("3. Blackwell 算力验证:", torch.cuda.get_device_capability(0))