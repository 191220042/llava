from hmquant.ptq.hm_fp8 import fp16_to_e5m8
import torch
import torch.nn as nn
from torch import Tensor
CLIPMIN = 1e-5
def my_pre_forward_hook(module,input):
    temp_input = input[0]
    temp_input = fp16_to_e5m8(temp_input,dim=-1,nshare=64)
    res_input = tuple([temp_input])
    return res_input


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def change_n_bits( n_bits, disable_zero_point):
    if disable_zero_point:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** (n_bits) - 1
    return qmin, qmax

def pergroup_quantaffine(x, scale, round_zero_point,group_size,n_bits):
    if len(x.shape)>1:
        dim1, dim2 = x.shape
        dim_flag = 2
    else:
        dim1 = x.shape
        dim_flag = 1
    x = x.reshape(-1, group_size)
    x_int = round_ste(x / scale)
    if round_zero_point is not None:
        x_int = x_int.add(round_zero_point)
    qmin,qmax = change_n_bits(n_bits, disable_zero_point=False)
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = x_int
    if round_zero_point is not None:
        x_dequant = x_dequant.sub(round_zero_point)
    x_dequant = x_dequant.mul(scale)
    if group_size:
        if dim_flag==2:
            x_dequant = x_dequant.reshape(dim1, dim2)
        else:
            x_dequant = x_dequant.reshape(dim1)
    return x_dequant

def per_token_dynamic_calibration(x, group_size, n_bits):
    x = x.reshape(-1,group_size)
    reduce_shape = [-1]
    xmin = x.amin(reduce_shape, keepdim=True)
    xmax =  x.amax(reduce_shape, keepdim=True)
    # symmetric
    abs_max = torch.max(xmax.abs(),xmin.abs())
    scale = abs_max / (2**(n_bits-1)-1)
    scale = scale.clamp(min=CLIPMIN, max=1e4)
    zero_point = (2**(n_bits-1)-1)*torch.ones_like(scale)
    # disable zero point=False
    round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    return scale,round_zero_point


from hm_sefp.hmquant.ptq.sefp.hm_sefp import conv_hm_fp_quant_forward,set_hmfp_dict
HMFP_DICT = dict()
HMFP_DICT['hardware_align'] = True
HMFP_DICT['mode'] = 'sefp'
HMFP_DICT['psum_dtype'] = 'fp32'
HMFP_DICT['extra_gpu'] = True
# HMFP_DICT['psum_round_mode']='rne'
HMFP_DICT['conv'] = dict(
    act=dict(type="fp16_to_e5m7", nshare=64),
    weight=dict(type="fp16_to_e5m7", nshare=64,)
    )
HMFP_DICT['matmul'] = dict(
    left=dict(type="fp16_to_e5m7", nshare=64, ),
    right=dict(type="fp16_to_e5m7", nshare=64,),
    )
set_hmfp_dict(HMFP_DICT)


def wrap_model(model:nn.Module):
    class WrapperConv(nn.Module):
        def __init__(self,model):
            super(WrapperConv,self).__init__()
            self.model = model
            self.weight = model.weight
            
        def forward(self, input: Tensor):
            result = conv_hm_fp_quant_forward(self.model,input)
            return result   

    wrap_model = WrapperConv(model)   
    return wrap_model      

def get_model(model):
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) or isinstance(module,nn.Conv2d):
            parent_name = name.rsplit(".",1)[0] if '.' in name else ''
            parent = dict(model.named_modules())[parent_name]
            setattr(parent,name.split('.')[-1],wrap_model(module))

    # for name,module in model.named_modules():
    #     if isinstance(module,nn.Conv2d):
    #         parent_name = name.rsplit(".",1)[0] if '.' in name else ''
    #         parent = dict(model.named_modules())[parent_name]
    #         setattr(parent,name.split('.')[-1],wrap_model(module))
    
    # for layer in model.model.layers:
        # layer.self_attn.q_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.k_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.v_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.o_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.gate_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.up_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.down_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.act_fn.register_forward_pre_hook(my_pre_forward_hook)
        # layer.input_layernorm.register_forward_pre_hook(my_pre_forward_hook)
        # layer.post_attention_layernorm.register_forward_pre_hook(my_pre_forward_hook)

    # for layer in model.model.vision_tower.vision_tower.vision_model.encoder.layers:
        # layer.self_attn.q_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.k_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.v_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.self_attn.out_proj.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.fc1.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.fc2.register_forward_pre_hook(my_pre_forward_hook)
        # layer.mlp.activation_fn.register_forward_pre_hook(my_pre_forward_hook)
        # layer.layer_norm1.register_forward_pre_hook(my_pre_forward_hook)
        # layer.layer_norm2.register_forward_pre_hook(my_pre_forward_hook)
        
    # model.model.vision_tower.vision_tower.vision_model.post_layernorm.register_forward_pre_hook(my_pre_forward_hook)
    # model.model.vision_tower.vision_tower.vision_model.pre_layrnorm.register_forward_pre_hook(my_pre_forward_hook)
    
    # for layer in model.model.mm_projector:
            
    #     layer.register_forward_pre_hook(my_pre_forward_hook)

    return model
