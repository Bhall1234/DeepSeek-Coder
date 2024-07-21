from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)


#1.3b parameters | Setup with 1 node, 1 GPU per node.
"""
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 1280M total params, 66M largest layer params.
  per CPU  |  per GPU |   Options
   32.20GB |   0.25GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   32.20GB |   0.25GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
   28.62GB |   2.63GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   28.62GB |   2.63GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    0.37GB |  21.71GB | offload_param=none, offload_optimizer=none, zero_init=1
    7.15GB |  21.71GB | offload_param=none, offload_optimizer=none, zero_init=0
"""

#6.7b parameters | Setup with 1 node, 1 GPU per node.
"""
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 6608M total params, 132M largest layer params.
  per CPU  |  per GPU |   Options
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
  147.71GB |  12.80GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  147.71GB |  12.80GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    0.74GB | 111.27GB | offload_param=none, offload_optimizer=none, zero_init=1
   36.93GB | 111.27GB | offload_param=none, offload_optimizer=none, zero_init=0
"""

#6.7b parameters | Setup with 1 node, 4 GPU per node.
"""
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 4 GPUs per node.
SW: Model with 6608M total params, 132M largest layer params.
  per CPU  |  per GPU |   Options
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
  147.71GB |   3.57GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  147.71GB |   3.57GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    2.95GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=1
  147.71GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=0
"""

#6.7b parameters | Setup with 1 node, 2 GPU per node.
"""
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 2 GPUs per node.
SW: Model with 6608M total params, 132M largest layer params.
  per CPU  |  per GPU |   Options
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  166.17GB |   0.49GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
  147.71GB |   6.65GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  147.71GB |   6.65GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    1.48GB |  55.88GB | offload_param=none, offload_optimizer=none, zero_init=1
   73.85GB |  55.88GB | offload_param=none, offload_optimizer=none, zero_init=0
"""


# http://www.georgesung.com/ai/qlora-ift/
# https://colab.research.google.com/drive/1IlpeofYD9EU6dNHyKKObZhIzkBMyqlUS