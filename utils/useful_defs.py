import torch
import numpy as np
from packaging import version
import transformers

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # removing position_ids to maintain compatibility with latest transformers update        
        if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]
    # for k in state_dict:
    #     if k.startswith('transformer'):
    #         v = state_dict.pop(k)
    #         state_dict['text_branch.' + k[12:]] = v
    return state_dict