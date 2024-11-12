import torch

is_gpu = torch.cuda.is_available()
devices_amount = torch.cuda.device_count()
current_device = torch.cuda.get_device_name(0)

if is_gpu:
    print('GPU is available')
    print("Num GPUs Available:", devices_amount)
    print("Device type:", current_device)
else:
    print('CPU is available')
