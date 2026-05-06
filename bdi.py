

import torch
import torch.nn as nn
def blockdiag_init(self, adapter_name, init_lora_weights):
    weight = self.get_base_layer().weight
    dtype = weight.dtype
    lora_A = self.lora_A[adapter_name].weight  # [r, in_features]
    r, in_features = lora_A.shape

    with torch.no_grad():
        lora_A.zero_()
        for start in range(0, in_features, r):
            end = min(start + r, in_features)
            size = end - start
            lora_A[:size, start:end] = torch.eye(size, dtype=dtype, device=lora_A.device)

    lora_B = torch.zeros_like(self.lora_B[adapter_name].weight)
    self.lora_B[adapter_name].weight = nn.Parameter(lora_B.contiguous().to(dtype))