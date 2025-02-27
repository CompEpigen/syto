import numpy as np
import torch
def count_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def calculate_batch_size(gb_per_seq:int,cpu_batch_size:int, utilization_coeff:float=0.85):
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        recomended_batch_size = int((vram*utilization_coeff/gb_per_seq)//10*10)
    else:
        recomended_batch_size = cpu_batch_size # assuming at least 16 gb of RAM if run on CPU
    return recomended_batch_size