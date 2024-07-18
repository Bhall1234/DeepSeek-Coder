from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)


"""
/home/ben/Desktop/DeepSeek-Coder/.venv/lib/python3.12/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
[2024-07-18 11:07:39,584] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/home/ben/Desktop/DeepSeek-Coder/.venv/lib/python3.12/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
/home/ben/Desktop/DeepSeek-Coder/.venv/lib/python3.12/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
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