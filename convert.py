import onnx
import torch
from onnx2torch import convert
from torch import nn

# Path to ONNX model
onnx_model_path = "face_detection_yunet_2023mar.onnx"
# You can pass the path to the onnx model to convert it or...
model = convert(onnx_model_path)
dummy = torch.zeros(1, 3, 320, 320)

with torch.no_grad():
    outputs = model(dummy)
bn_ma_params = 0
bn_gd_params = 0
conv_params = 0
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        bn_ma_params += m.running_mean.numel()
        bn_ma_params += m.running_var.numel()
        bn_gd_params += sum([p.numel() for p in m.parameters()])
    if isinstance(m, (nn.Conv2d)):
        conv_params += sum([p.numel() for p in m.parameters()])
        
print(f"BatchNorm moving-average params: {bn_ma_params}")
print(f"BatchNorm GD params: {bn_gd_params}")
print(f"Conv params: {conv_params}")
print(f"Total GD params {conv_params + bn_gd_params}")
print(f"Total params {conv_params + bn_gd_params + bn_ma_params}")
