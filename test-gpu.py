import torch
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
print(device)
print(f"{'='*20} Check cuda availability {'='*20}")

print(torch.cuda.is_available())

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()

    if num_gpus > 0:
        for gpu in range(num_gpus):
            print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
    else:
        print("No gpu")
else:
    print("cuda not available")
    
print(f"{'='*20} end check cuda availability {'='*20}")