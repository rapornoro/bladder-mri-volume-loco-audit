"""Post hoc analysis: 3D ResNet-18 initialised with MedicalNet weights (MONAI pretrained=True, 23 medical datasets),
conv1 expanded to 2 channels (image channel = pretrained kernel, mask channel = small random), fine-tuned end to end.
Same crops, augmentation, fixed 40-epoch schedule, LOCO folds, 3 seeds + 5 inner OOF models as s04 (lr 1e-4).
Output: work/cnn_mednet/loco_c*.npz"""
import os
import sys
import torch
import torch.nn as nn
from monai.networks.nets import resnet18

sys.path.insert(0, os.path.dirname(__file__))
import s04_cnn3d as base  # noqa: E402


def mednet_model():
    m = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1, pretrained=True, feed_forward=False,
                 shortcut_type="A", bias_downsample=True)
    w = m.conv1.weight.data  # 64,1,7,7,7
    conv = nn.Conv3d(2, w.shape[0], kernel_size=m.conv1.kernel_size, stride=m.conv1.stride,
                     padding=m.conv1.padding, bias=m.conv1.bias is not None)
    conv.weight.data[:, :1] = w
    conv.weight.data[:, 1:] = torch.randn_like(w) * w.std() * 0.1
    m.conv1 = conv
    m.fc = nn.Linear(512, 1)
    return m.to(base.DEV)


base.new_model = mednet_model
base.LR = 1e-4
base.OUT = os.path.join(base.WORK, "cnn_mednet")
os.makedirs(base.OUT, exist_ok=True)

if __name__ == "__main__":
    base.run_loco()
