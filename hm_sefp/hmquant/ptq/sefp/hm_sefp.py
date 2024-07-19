
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

from hm_sefp.hmquant.ptq.sefp.hm_fp8 import to_hmfp16, to_hmfp8, fp16_to_e5m4, fp16_to_e5m8
from hm_sefp.hmquant.ptq.sefp.hm_fp8 import fp16_to_e5m10, fp16_to_e5m12, keep_fp16, to_share_exp_wrapper
from hm_sefp.hmquant.ptq.sefp.lut import cut_points_dict
from hm_sefp.hmquant.ptq.sefp.lut import NewTable

HMFP_DICT = dict()

def set_hmfp_dict(hm_fp_set):
    global HMFP_DICT
    HMFP_DICT = hm_fp_set

def get_hmfp_dict():
    global HMFP_DICT
    return HMFP_DICT

__all__ = ['conv_hm_fp_quant_forward',
           'matmul_hm_fp_quant_forward',
           'deconv_hm_fp_quant_forward',
           'add_hm_fp_quant_forward',
           'sub_hm_fp_quant_forward',
           'mul_hm_fp_quant_forward',
           'lut_hm_fp_quant_forward',
           'softmax_hm_fp_quant_forward',
           'gn_hm_fp_quant_forward',
           'ln_hm_fp_quant_forward',
           'set_hmfp_dict',
           'get_hmfp_dict',
           ]
        
def check_is_float16(x):
    if x is not None and isinstance(x, torch.Tensor):
        if x.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float16]:
            print("x dtype: {}".format(x.dtype))
            x = x.to(torch.float16)
    return x

def quant_act(act, we_dict, unfold_x, hardware_align, extra_gpu, dim=-1):
    if act.get("type") == "keep_fp16":
        unfold_x = keep_fp16(unfold_x, dim=dim, nshare=act.get("nshare"))
        return unfold_x
    elif act.get("type").startswith("fp16_to_e"):
        config = act.get("type").split("fp16_to_e")[1]
        config = config.split("m")
        exp = int(config[0])
        mas = int(config[1])
        rounding = act.get("rounding", 'rne')
        keep_bit_for_exp = act.get("keep_bit_for_exp", True)
        nshare=act.get("nshare")
        if we_dict.get("type") == "keep_fp16" or not hardware_align:
            unfold_x = to_share_exp_wrapper(
                unfold_x, dim=dim, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=False, assemble=True, dtype=torch.float32, extra_gpu=extra_gpu)
            return unfold_x
        else:
            unfold_x_int, unfold_x_scale = to_share_exp_wrapper(
                unfold_x, dim=dim, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=True, assemble=False, dtype=torch.float32, extra_gpu=extra_gpu)
            unfold_x_scale.exp2_()
            return unfold_x_int, unfold_x_scale,  mas + 1
    else:
        raise NotImplementedError(f"act type {act} not support")

def quant_weight(act, we_dict, self, hardware_align, extra_gpu, mode, dim=0, w=None):
    if w is None:
        w = self.weight.flatten(1)
        w = w.t()

    nshare = we_dict.get("nshare", 64)
    if we_dict.get("type") == "keep_fp16":
        w = keep_fp16(w, dim=dim, nshare=nshare)
        return w
    elif we_dict.get("type").startswith("fp16_to_e") and mode in ['sefp']:
        config = we_dict.get("type").split("fp16_to_e")[1]
        config = config.split("m")
        exp = int(config[0])
        mas = int(config[1])
        rounding = we_dict.get("rounding", 'rne')        
        keep_bit_for_exp = we_dict.get("keep_bit_for_exp", True)
        weight_bit = mas + 1
        if act.get("type") == "keep_fp16" or not hardware_align:
            w = to_share_exp_wrapper(
                w, dim=dim, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=False, assemble=True, dtype=torch.float32, extra_gpu=extra_gpu)
            return w
        else:
            w_int, w_scale = to_share_exp_wrapper(
                w, dim=dim, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=True, assemble=False, dtype=torch.float32, extra_gpu=extra_gpu)
            w_scale.exp2_()
            return w_int, w_scale, weight_bit
    elif mode in ['ssfp'] and hasattr(self, 'w_quant_param'):        
        if self.w_quant_param is None or ',' in self.w_quant_param.granularity:
            from hmquant.observers import build_observer
            w_observer = build_observer(cfg="minmax", granularity="dim0")
            w_observer(self.weight)
            w_quant_param = w_observer.calculate_qparams(self.w_dtype)
        else:
            w_quant_param = self.w_quant_param
        
        op_class = getattr(self, 'op_class', '')
        if op_class in ['ConvTransposed2d', 'CIMDTransposeConv2d']:
            w_int = w
        else:
            w_int = w_quant_param.quant_tensor(self.weight, False)
            w_int = w_int.to(torch.float32)
            w_int = w_int.flatten(1)
            w_int = w_int.t()
        if w_int.shape[0] % nshare != 0:
            pad = nshare - (w_int.shape[0] % nshare)
            pad_val = torch.zeros([pad, w_int.shape[1]], dtype=w_int.dtype, device=w_int.device)
            w_int = torch.concat([w_int, pad_val], dim=0)
        w_int = w_int.reshape(-1, nshare, w_int.shape[1])
        
        if op_class in ['ConvTransposed2d', 'CIMDTransposeConv2d']:
            from hmquant.observers import build_observer
            if False:
                w_observer = build_observer(cfg="minmax", granularity=['dim0', 'dim2'])
                w_observer(w_int)
                w_quant_param = w_observer.calculate_qparams(self.w_dtype)
                w_int = w_quant_param.quant_tensor(w_int, False)
                w_scale = w_quant_param.scale
                w_scale = w_scale.reshape(int(w_int.shape[0]), 1, -1)
            else:
                w_observer = build_observer(cfg="minmax", granularity='dim1')
                w_int = w_int.reshape(-1, w_int.shape[-1])
                w_observer(w_int)
                w_quant_param = w_observer.calculate_qparams(self.w_dtype)
                w_int = w_quant_param.quant_tensor(w_int, False)
                w_int = w_int.reshape(-1, nshare, w_int.shape[1])
                w_scale = w_quant_param.scale.repeat(int(w_int.shape[0]))
                w_scale = w_scale.reshape(int(w_int.shape[0]), 1, -1)
            w_int = w_int.to(torch.float32)           
        else:
            w_scale = w_quant_param.scale.repeat(int(w_int.shape[0]))
            w_scale = w_scale.reshape(int(w_int.shape[0]), 1, -1)
        weight_bit = w_quant_param.bitwidth
        return w_int, w_scale, weight_bit
    elif mode in ['ssfp'] and hasattr(self, 'quant_weight'):
        w = self.weight.flatten(1)
        w = w.t()

        w_int = self.quant_weight
        w_int = w_int.to(torch.float32)
        w_int = w_int.flatten(1)
        w_int = w_int.t()

        if w_int.shape[0] % nshare != 0:
            pad = nshare - (w_int.shape[0] % nshare)
            pad_val = torch.zeros([pad, w_int.shape[1]], dtype=w_int.dtype, device=w_int.device)
            w_int = torch.concat([w_int, pad_val], dim=0)
            w = torch.concat([w, pad_val], dim=0)
        w_int = w_int.reshape(-1, nshare, w_int.shape[1])
        w = w.reshape(-1, nshare, w.shape[1])

        w_scale = w.div(w_int)
        w_scale.masked_fill_(w_int == 0, 0)
        w_scale, _ = w_scale.max(dim=1, keepdim=True)
        weight_bit = w_int.abs().max().log2().ceil().item()
        weight_bit = int(weight_bit) + 1
        return w_int, w_scale, weight_bit
    else:
        raise NotImplementedError("weight type not support")

def conv_hm_fp_quant_forward(self, x):
    #return self.raw_forward(x)
    hmfp_dict = get_hmfp_dict()
    hmfp_conv_dict = hmfp_dict.get("conv")
    act = hmfp_conv_dict.get("act")
    we_dict = hmfp_conv_dict.get("weight")
    hardware_align = hmfp_dict.get('hardware_align', False)
    mode = hmfp_dict.get('mode', 'sefp').lower()
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')
    extra_gpu = hmfp_dict.get('extra_gpu', False)
    x = check_is_float16(x)
    
    if hasattr(self, 'extra_padding') and self.extra_padding is not None:
        x = F.pad(x, self.extra_padding)

    if hasattr(self, 'groups') and self.groups != 1 and hasattr(self, 'transform_grouped_conv'):
        self.transform_grouped_conv()
            
    if isinstance(self, nn.Linear):
        unfold_x = x
    else:
        unfold_x = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        unfold_x = unfold_x.transpose(1, 2)
    
    nshare = act.get("nshare", 64)
    assert nshare == we_dict.get("nshare", 64), "two input tensors should have same nshare"

    val = quant_act(act, we_dict, unfold_x, hardware_align, extra_gpu, dim=-1)
    if isinstance(val, tuple):
        unfold_x_int, unfold_x_scale, act_bit = val
    else:
        unfold_x = val

    if hasattr(self, '_quant_weight_val_') and self._quant_weight_val_ is not None:
        val = self._quant_weight_val_
    else:
        val = quant_weight(act, we_dict, self, hardware_align, extra_gpu, mode, dim=0)
        setattr(self, '_quant_weight_val_', val)
    if isinstance(val, tuple):
        w_int, w_scale, weight_bit = val
    else:
        w = val

    if we_dict.get("type") == "keep_fp16" or act.get("type") == "keep_fp16" or not hardware_align:
        if mode in ['ssfp'] and we_dict.get("type") != "keep_fp16":
            w = w_int.mul_(w_scale)
            w_int = w_scale = None
            w = w.reshape(-1, int(w.shape[-1]))
            if w.shape[0] != unfold_x.shape[-1]: # remove pad
                w = w[:int(unfold_x.shape[-1])]
        unfold_x = unfold_x.to(torch.float32)
        w = w.to(torch.float32)
        out = torch.matmul(unfold_x, w)
        unfold_x = w = None
    else:        
        unfold_x = None
        if extra_gpu:
            unfold_x_int = unfold_x_int.to('cuda:1')
            unfold_x_scale = unfold_x_scale.to('cuda:1')
            w_int = w_int.to("cuda:1")
            w_scale = w_scale.to("cuda:1")

        groups = int(w_int.shape[0])
        unfold_x_int = unfold_x_int.reshape(-1, groups, nshare)
        unfold_x_scale = unfold_x_scale.reshape(-1, groups, 1).to(torch.float16)
        out = None
        for i in range(groups):
            fm = unfold_x_int[:,i]
            wt = w_int[i]
            if (act_bit + weight_bit + np.log2(nshare)) > 25: # float64
                fm = fm.to(dtype=torch.float64)
                wt = wt.to(dtype=torch.float64)
            psum = torch.matmul(fm, wt)
            fm = wt = None
            psum = psum.to(dtype=torch.float32)

            psum.mul_(unfold_x_scale[:,i])
            psum.mul_(w_scale[i].to(torch.float16))

            if psum_dtype == 'fp24':
                from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                psum = fp32_to_fp24(psum, round_mode=psum_round_mode)

            if out is None:
                out = psum
            else:
                out.add_(psum)
                if psum_dtype == 'fp24':
                    from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                    out = fp32_to_fp24(out, round_mode=psum_round_mode)
                    
    if True:
        if not isinstance(self, nn.Linear):
            #assert min(self.padding) == max(self.padding)
            #assert min(self.kernel_size) == max(self.kernel_size)
            #assert min(self.dilation) == max(self.dilation)
            #assert min(self.stride) == max(self.stride)
            out_shape_h = x.shape[2] + 2 * self.padding[0] - \
                self.dilation[0] * (self.kernel_size[0] - 1) - 1
            out_shape_h = out_shape_h / self.stride[0] + 1
            b, _, _, _ = x.shape
            c = self.out_channels
            h = int(out_shape_h)
            out = out.reshape(b, -1, c)
            out = out.transpose(1, 2)
            out = out.view(b, c, h, -1)

        else:
            if len(x.shape) == 3:
                b, t, _ = x.shape
                out = out.reshape(b, t, -1)

    out.clamp_(max=65504, min=-65504)
    out = out.to(torch.float16)

    if out.device != x.device:
        out = out.to(x.device)

    if self.bias is not None:
        if isinstance(self, nn.Linear):
            bias = self.bias.view(1, -1)
        else:
            bias = self.bias.view(1, -1, 1, 1)
        bias = bias.to(torch.float16)
        out.add_(bias)

    if hasattr(self, 'with_relu') and self.with_relu:
        out = F.relu(out)
    
    check_is_float16(out)
    return out

def matmul_hm_fp_quant_forward(self, x1, x2, transpose=False):
    #return self.raw_forward(x1, x2)
    hmfp_dict = get_hmfp_dict()
    hardware_align = hmfp_dict.get('hardware_align', False)
    hmfp_matmul_dict = hmfp_dict.get("matmul")
    left = hmfp_matmul_dict.get("left")
    right = hmfp_matmul_dict.get("right")
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')
    extra_gpu = hmfp_dict.get('extra_gpu', False)
    check_is_float16(x1)
    check_is_float16(x2)
    device = x1.device
    
    nshare = left.get("nshare", 64)
    assert nshare == right.get("nshare", 64), "two input tensors should have same nshare"

    if left.get("type") == "keep_fp16":
        x1 = keep_fp16(x1, dim=-1, nshare=left.get("nshare"))
    elif left.get("type").startswith("fp16_to_e"):
        config = left.get("type").split("fp16_to_e")[1]
        config = config.split("m")
        exp = int(config[0])
        mas = int(config[1])
        rounding = left.get("rounding", 'rne')     
        keep_bit_for_exp = left.get("keep_bit_for_exp", True)   
        left_bit = mas + 1
        if right.get("type") == "keep_fp16" or not hardware_align:
            x1 = to_share_exp_wrapper(
                x1, dim=-1, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=False, assemble=True, dtype=torch.float32, extra_gpu=extra_gpu)
        else:
            x1_int, x1_scale = to_share_exp_wrapper(
                x1, dim=-1, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=True, assemble=False, dtype=torch.float32, extra_gpu=extra_gpu)            
    else:
        raise NotImplementedError("left type not support")

    if transpose:
        x2 = x2.transpose(1, 2)

    if right.get("type") == "keep_fp16":
        x2 = keep_fp16(x2, dim=-2, nshare=right.get("nshare"))
    elif right.get("type").startswith("fp16_to_e"):
        config = right.get("type").split("fp16_to_e")[1]
        config = config.split("m")
        exp = int(config[0])
        mas = int(config[1])
        rounding = right.get("rounding", 'rne')
        keep_bit_for_exp = right.get("keep_bit_for_exp", True)   
        nshare = right.get("nshare")
        right_bit = mas + 1
        if left.get("type") == "keep_fp16" or not hardware_align:
            x2 = to_share_exp_wrapper(
                x2, dim=-2, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=False, assemble=True, dtype=torch.float32, extra_gpu=extra_gpu)
        else:
            x2_int, x2_scale = to_share_exp_wrapper(
                x2, dim=-2, exp=exp, mas=mas, nshare=nshare, rounding=rounding, keep_bit_for_exp=keep_bit_for_exp,
                return_padded=True, assemble=False, dtype=torch.float32, extra_gpu=extra_gpu)
    else:
        raise NotImplementedError(f"right type {right} not support")

    if True:
        if right.get("type") == "keep_fp16" or left.get("type") == "keep_fp16" or not hardware_align:            
            if extra_gpu:
                x1 = x1.to("cuda:1").to(torch.float32)
                x2 = x2.to("cuda:1").to(torch.float32)
            else:
                x1 = x1.to(torch.float32)
                x2 = x2.to(torch.float32)
            out = torch.matmul(x1, x2)
            x1 = x2 = None
        else:
            if extra_gpu:
                x1_scale = x1_scale.to("cuda:1")
                x2_scale = x2_scale.to("cuda:1")
                x1_int = x1_int.to("cuda:1")
                x2_int = x2_int.to("cuda:1")

            x1_scale.exp2_()
            x2_scale.exp2_()

            groups = int(x1_int.shape[-2])
            if len(x1.shape) == 3:
                x1_int = x1_int.reshape(int(x1.shape[0]), -1, groups, nshare)
                x1_scale = x1_scale.reshape(int(x1.shape[0]), -1, groups, 1)
            elif len(x1.shape) == 4:
                x1_int = x1_int.reshape(int(x1.shape[0]), int(x1.shape[1]), -1, groups, nshare)
                x2_int = x2_int.reshape(int(x2.shape[0]), int(x2.shape[1]), groups, nshare, -1)
            else:
                x1_int = x1_int.reshape(-1, groups, nshare)
                x1_scale = x1_scale.reshape(-1, groups, 1)

            out = None
            for i in range(groups):
                if len(x2.shape) == 3:
                    part_x2_int = x2_int[:,i]
                    part_x2_scale = x2_scale[:,i]
                elif len(x2.shape) == 4:
                    part_x2_int = x2_int[:,:,i]
                    part_x2_scale = x2_scale[:,:,i]
                else:
                    part_x2_int = x2_int[i]
                    part_x2_scale = x2_scale[i]
                    
                if len(x1.shape) == 3:       
                    part_x1_int = x1_int[:,:,i]
                    part_x1_scale = x1_scale[:,:,i]      
                elif len(x1.shape) == 4:
                    part_x1_int = x1_int[:,:,:,i]
                    part_x1_scale = x1_scale[:,:,:,i]                    
                else:                    
                    part_x1_int = x1_int[:,i]
                    part_x1_scale = x1_scale[:,i]
            
                if right_bit + left_bit + np.log2(nshare) > 25:
                    part_x1_int = part_x1_int.to(dtype=torch.float64)
                    part_x2_int = part_x2_int.to(dtype=torch.float64)
                    psum = torch.matmul(part_x1_int, part_x2_int)
                    psum = psum.to(torch.float32)
                else:
                    psum = torch.matmul(part_x1_int, part_x2_int)
                psum.mul_(part_x1_scale.to(torch.float16))
                psum.mul_(part_x2_scale.to(torch.float16))

                if psum_dtype == 'fp24':
                    from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                    psum = fp32_to_fp24(psum, round_mode=psum_round_mode)

                if out is None:
                    out = psum
                else:
                    out.add_(psum)
                    if psum_dtype == 'fp24':
                        from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                        out = fp32_to_fp24(out, round_mode=psum_round_mode)

            x1_int = x2_int = psum = None
            x1_scale = x2_scale = None
            part_x2_int = part_x2_int = part_x1_scale = part_x2_scale = None
            
        out.clamp_(max=65504, min=-65504)
        out = out.to(torch.float16)
        
        if out.device != device:
            out = out.to(device)

        if hasattr(self, 'scale') and self.scale is not None:
            if isinstance(self.scale, torch.Tensor):
                scale = self.scale.to(torch.float16)
            else:
                scale = torch.Tensor([self.scale]).to(dtype=torch.float16, device=out.device)
            out = out.mul_(scale)
    
    check_is_float16(out)
    return out

def deconv_hm_fp_quant_forward(self, input):
    hmfp_dict = get_hmfp_dict()
    hmfp_conv_dict = hmfp_dict.get("conv")
    act = hmfp_conv_dict.get("act")
    we_dict = hmfp_conv_dict.get("weight")
    hardware_align = hmfp_dict.get('hardware_align', False)
    mode = hmfp_dict.get('mode', 'sefp').lower()
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')
    extra_gpu = hmfp_dict.get('extra_gpu', False)
    check_is_float16(input)

    ####
    op_class = getattr(self, 'op_class', 'CIMDTransposeConv2d')
    if hasattr(self, 's') and op_class == 'CIMDTransposeConv2d':
        stride = self.s
    elif hasattr(self, 'ori_stride'):
        stride = self.ori_stride
    elif hasattr(self, 'stride'):
        stride = self.stride
    else:
        raise RuntimeError("no stride")
    if isinstance(stride, tuple) and len(stride) == 2:
        assert stride[0] == stride[1], "invalid stride"
        stride = stride[0]
    
    if hasattr(self, 'ori_padding') and op_class == 'ConvTransposed2d':
        padding = self.ori_padding
    elif hasattr(self, 'padding'):
        padding = self.padding
    else:
        raise RuntimeError("no padding")
    if isinstance(padding, tuple) and len(padding) == 2:
        assert padding[0] == padding[1], "invalid padding"
        padding = padding[0]
    
    if hasattr(self, 'raw_weight'):
        weight = self.raw_weight
    elif hasattr(self, 'weight'):
        weight = self.weight
    else:
        raise RuntimeError("no weight")
    
    if hasattr(self, 'raw_bias') and op_class == 'CIMDTransposeConv2d':
        bias = self.raw_bias
    elif hasattr(self, 'bias'):
        bias = self.bias
    else:
        raise RuntimeError("no bias")
    ####
    
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    
    assert len(weight.shape) == 4 and weight.shape[-1] == weight.shape[-2], 'invalid kernel_size'
    kernel_size = weight.shape[-1]
    co = weight.shape[1]
    bs, ci, h, w = input.shape

    if stride > 1:
        x = torch.arange(0, w)
        y = torch.arange(0, h)
        grid_h, grid_w = torch.meshgrid(y, x, indexing='ij')
        pool_size = torch.Size([bs, ci, (h-1)*stride + 1, (w-1)*stride + 1])
        indices = grid_w * stride + grid_h * pool_size[-1] * stride
        indices = indices.to(input.device)
        pool_indices = indices.repeat(bs, ci, 1, 1)
        unpool = nn.MaxUnpool2d(stride, stride=stride)
        output = unpool(input, pool_indices, output_size=pool_size)
        unfold = F.unfold(output, (kernel_size, kernel_size), padding=(kernel_size-1-padding, kernel_size-1-padding))
        unfold = unfold.transpose(1, 2)
    else:
        raise RuntimeError("to be support, but easy to support")
    
    nshare = act.get("nshare", 64)
    assert nshare == we_dict.get("nshare", 64), "two input tensors should have same nshare"
    
    if hasattr(self, 'conv_weight') and self.conv_weight is not None:
        conv_weight = self.conv_weight
    else:
        conv_weight = weight.transpose(0, 1)
        conv_weight = torch.rot90(conv_weight, 2, [2,3])
        conv_weight = conv_weight.reshape(conv_weight.size(0), -1)
        conv_weight = conv_weight.transpose(0, 1)
        setattr(self, 'conv_weight', conv_weight)
    
    ### quant begin
    val = quant_act(act, we_dict, unfold, hardware_align, extra_gpu, dim=-1)
    if isinstance(val, tuple):
        unfold_x_int, unfold_x_scale, act_bit = val
    else:
        unfold = val

    #assert we_dict.get("type") == "keep_fp16" or mode not in ['ssfp'], "mode not support yet"
    if hasattr(self, '_quant_weight_val_') and self._quant_weight_val_ is not None:
        val = self._quant_weight_val_
    else:
        val = quant_weight(act, we_dict, self, hardware_align, extra_gpu, mode, dim=0, w=conv_weight)
        setattr(self, '_quant_weight_val_', val)
    if isinstance(val, tuple):
        w_int, w_scale, weight_bit = val
    else:
        conv_weight = val
    ### quant end

    if we_dict.get("type") == "keep_fp16" or act.get("type") == "keep_fp16" or not hardware_align:
        conv_output = torch.matmul(unfold, conv_weight)
    else:
        groups = int(w_int.shape[0])
        unfold_x_int = unfold_x_int.reshape(-1, groups, nshare)
        unfold_x_scale = unfold_x_scale.reshape(-1, groups, 1)
        out = None
        for i in range(groups):
            fm = unfold_x_int[:,i]
            wt = w_int[i]
            if (act_bit + weight_bit + np.log2(nshare)) > 25: # float64
                fm = fm.to(dtype=torch.float64)
                wt = wt.to(dtype=torch.float64)
            psum = torch.matmul(fm, wt)
            fm = wt = None
            psum = psum.to(dtype=torch.float32)

            psum.mul_(unfold_x_scale[:,i].to(torch.float16))
            psum.mul_(w_scale[i].to(torch.float16))

            if psum_dtype == 'fp24':
                from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                psum = fp32_to_fp24(psum, round_mode=psum_round_mode)

            if out is None:
                out = psum
            else:
                out.add_(psum)
                if psum_dtype == 'fp24':
                    from hm_sefp.hmquant.ptq.sefp.fp24 import fp32_to_fp24
                    out = fp32_to_fp24(out, round_mode=psum_round_mode)
        conv_output = out.reshape(bs, -1, out.shape[-1])
        out = psum = None

    conv_output = conv_output.transpose(1, 2)
    oh = pool_size[-2] + 2 * (kernel_size-1-padding) - (kernel_size-1)
    ow = pool_size[-1] + (kernel_size-1) - 2*padding
    out = conv_output.reshape(bs, co, oh, ow)
    
    out.clamp_(max=65504, min=-65504)
    out = out.to(torch.float16)

    if out.device != input.device:
        out = out.to(input.device)

    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    
    # torch_out = F.conv_transpose2d(input, weight, bias, stride=stride, padding=padding)
    # print("Error: {}".format((torch_out - out).abs().max().item()))
    check_is_float16(out)
    return out

def add_hm_fp_quant_forward(self, x1, x2):
    x1 = check_is_float16(x1)
    x2 = check_is_float16(x2)

    if isinstance(x2, (float, int, bool)):
        x2 = torch.Tensor([x2]).to(device=x1.device, dtype=torch.float16)
    elif not isinstance(x2, torch.Tensor):
        print("x2 is not a Tensor or float number")

    y = x1 + x2
    check_is_float16(y)
    return y

def sub_hm_fp_quant_forward(self, x1, x2):
    x1 = check_is_float16(x1)
    x2 = check_is_float16(x2)

    if isinstance(x2, (float, int, bool)):
        x2 = torch.Tensor([x2]).to(device=x1.device, dtype=torch.float16)
    elif not isinstance(x2, torch.Tensor):
        print("x2 is not a Tensor or float number")

    y = x1 - x2
    check_is_float16(y)
    return y

def mul_hm_fp_quant_forward(self, x1, x2=None):
    x1 = check_is_float16(x1)
    x2 = check_is_float16(x2)

    if x2 is None:
        if hasattr(self, 'scale') and self.scale is not None:
            x2 = self.scale
        elif hasattr(self, 'scalar') and self.scalar is not None:
            x2 = self.scalar
        else:
            raise RuntimeError("x2 is invalid")

    if isinstance(x2, torch.Tensor):
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
    elif isinstance(x2, (float, int, bool)):
        x2 = torch.Tensor([x2]).to(device=x1.device, dtype=torch.float16)
    else:
        print("x2 is not a Tensor or float number")
    y = x1 * x2
    check_is_float16(y)
    return y

def lut_hm_fp_quant_forward(self, x, function=None):
    check_is_float16(x)
    
    if hasattr(self, 'function') and self.function is not None:
        function = self.function        
    assert function is not None and callable(function), "function is invaild"

    if not hasattr(self, 'table') or self.table is None or not isinstance(self.table, NewTable):
        setattr(self, 'table', None)
        cut_points = cut_points_dict[function.__name__]
        table_cut_points = [0] * 259 
        table_cut_points[0] = cut_points[0]
        table_cut_points[-1] = cut_points[-1]
        self.table = NewTable(function, cut_points[2:-2], table_size=257, min=cut_points[1], max=cut_points[-2], device=x.device)

        table_cut_points[1:-1] = self.table.index.tolist()
        self.table.index = torch.tensor(table_cut_points, dtype=torch.float16).to(x.device)
        self.table.table_size = 259
        self.table.table = self.table.func(self.table.index).clamp(-65504, 65504).to(torch.float16)      

    y = self.table(x)
    check_is_float16(y)
    return y

def softmax_hm_fp_quant_forward(self, x):
    check_is_float16(x)
    if not hasattr(self, 'lut_exp') or self.lut_exp is None or not isinstance(self.lut_exp, NewTable):
        setattr(self, 'lut_exp', None)
        cut_points = cut_points_dict["exp"]
        table_cut_points = [0] * 259 
        table_cut_points[0] = cut_points[0]
        table_cut_points[-1] = cut_points[-1]
        self.lut_exp = NewTable(torch.exp, cut_points[2:-2], table_size=257, min=cut_points[1], max=cut_points[-2], device=x.device)
    
        table_cut_points[1:-1] = self.lut_exp.index.tolist()
        self.lut_exp.index = torch.tensor(table_cut_points, dtype=torch.float16).to(x.device)
        self.lut_exp.table_size = 259
        self.lut_exp.table = self.lut_exp.func(self.lut_exp.index).clamp(-65504, 65504).to(torch.float16)    
    
    if not hasattr(self, 'lut_div') or self.lut_div is None or not isinstance(self.lut_div, NewTable):
        def divide(x):
            return 1 / x
        setattr(self, 'lut_div', None)
        cut_points = cut_points_dict["divide"]
        table_cut_points = [0] * 259 
        table_cut_points[0] = cut_points[0]
        table_cut_points[-1] = cut_points[-1]
        self.lut_div = NewTable(divide, cut_points[2:-2], table_size=257, min=cut_points[1], max=cut_points[-2], device=x.device)
    
        table_cut_points[1:-1] = self.lut_div.index.tolist()
        self.lut_div.index = torch.tensor(table_cut_points, dtype=torch.float16).to(x.device)
        self.lut_div.table_size = 259
        self.lut_div.table = self.lut_div.func(self.lut_div.index).clamp(-65504, 65504).to(torch.float16)
    
    axis = self.dim if hasattr(self, 'dim') else 1
    axis = self.axis if hasattr(self, 'axis') else axis
    re_x = x - x.max(dim=axis, keepdim=True)[0]
    exp_out = self.lut_exp(re_x, inplace=True)
    exp_sum = exp_out.sum(dim=axis, keepdim=True, dtype=torch.float16)
    exp_div = self.lut_div(exp_sum, inplace=True)
    y = exp_out.mul_(exp_div)
    check_is_float16(y)
    return y

def gn_hm_fp_quant_forward(self, x):
    check_is_float16(x)
    if self.weight is not None and isinstance(self.weight, torch.Tensor):
        self.weight = self.weight.to(torch.float16)
    if self.bias is not None and isinstance(self.bias, torch.Tensor):
        self.bias = self.bias.to(torch.float16)
    out = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    out = out.to(torch.float16)
    return out

def ln_hm_fp_quant_forward(self, x):
    check_is_float16(x)
    normalized_shape = []
    if hasattr(self, 'normalized_index'):
        normalized_shape = [x.size()[index] for index in self.normalized_index]    
    elif hasattr(self, 'normalized_shape'):
        normalized_shape = self.normalized_shape

    normalized_shape = list(normalized_shape)
    setattr(self, 'normalized_shape', normalized_shape)
    
    if list(self.weight.shape) != normalized_shape or list(self.bias.shape) != normalized_shape:
        if torch.all(self.weight == 1) and torch.all(self.bias == 0):
            self.weight.data = torch.ones(self.normalized_shape, dtype=torch.float16).to(x.device)
            self.bias.data = torch.zeros(self.normalized_shape, dtype=torch.float16).to(x.device)
        else:
            raise ValueError()
    
    if self.weight is not None and isinstance(self.weight, torch.Tensor):
        weight = self.weight.to(torch.float16)
    else:
        weight = None
    if self.bias is not None and isinstance(self.bias, torch.Tensor):
        bias = self.bias.to(torch.float16)
    else:
        bias = None

    out = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
    out = out.to(torch.float16)
    return out

def conv_hm_fp_get_hardware_params(self, param_dict):
    hmfp_dict = get_hmfp_dict()
    hmfp_conv_dict = hmfp_dict.get("conv")
    act = hmfp_conv_dict.get("act")
    we_dict = hmfp_conv_dict.get("weight")
    hardware_align = hmfp_dict.get('hardware_align', False)
    mode = hmfp_dict.get('mode', 'sefp').lower()
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')
    extra_gpu = hmfp_dict.get('extra_gpu', False)
    
    nshare = act.get("nshare", 64)
    assert nshare == we_dict.get("nshare", 64), "two input tensors should have same nshare"

    if hasattr(self, '_quant_weight_val_') and self._quant_weight_val_ is not None:
        val = self._quant_weight_val_
    else:
        val = quant_weight(act, we_dict, self, hardware_align, extra_gpu, mode, dim=0)
        setattr(self, '_quant_weight_val_', val)
    if isinstance(val, tuple):
        w_int, w_scale, weight_bit = val
    else:
        w = val

    param_dict["dilations"] = self.dilation
    param_dict["group"] = self.groups
    param_dict["kernel_shape"] = self.kernel_size
    param_dict["pads"] = self.padding
    param_dict["strides"] = self.stride

    param_dict['hmfp'] = mode

    config = act.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    mas = int(config[1])
    rounding = act.get("rounding", 'rne')
    keep_bit_for_exp = act.get("keep_bit_for_exp", True)    
    param_dict['hmfp_act_mas_bit'] = mas + 1
    param_dict['hmfp_act_exp_bit'] = exp
    param_dict['hmfp_act_rounding'] = rounding
    param_dict['hmfp_act_nshare'] = nshare
    param_dict['hmfp_act_keep_bit_for_exp'] = keep_bit_for_exp
    
    config = we_dict.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    rounding = we_dict.get("rounding", 'rne')
    keep_bit_for_exp = we_dict.get("keep_bit_for_exp", True)    
    param_dict['hmfp_weight_mas_bit'] = weight_bit
    param_dict['hmfp_weight_exp_bit'] = exp
    param_dict['hmfp_weight_rounding'] = rounding
    param_dict['hmfp_weight_nshare'] = nshare
    param_dict['hmfp_weight_keep_bit_for_exp'] = keep_bit_for_exp
    
    assert w_int.max() < pow(2, weight_bit-1) and w_int.min() >= -pow(2, weight_bit-1), "error data range"
    if weight_bit > 8:
        hmfp_weight = w_int.cpu().to(torch.int16).detach().numpy()
    else:
        hmfp_weight = w_int.cpu().to(torch.int8).detach().numpy()    
    param_dict["hmfp_weight"] = hmfp_weight
    param_dict["hmfp_weight_scale"] = w_scale.cpu().to(torch.float16).detach().numpy()
    
    param_dict['hmfp_psum_dtype'] = psum_dtype
    param_dict['hmfp_psum_round_mode'] = psum_round_mode
    
    param_dict["have_bias"] = True
    items = ["with_relu"]
    for item in items:
        if item not in param_dict and hasattr(self, item):
            param_dict[item] = getattr(self, item)

    param_dict["weight"] = self.weight.cpu().to(torch.float16).detach().numpy()
    if self.bias is not None:
        param_dict["bias"] = self.bias.cpu().to(torch.float16).detach().numpy()
    else:
        param_dict["bias"] = torch.zeros(self.weight.shape[0]).to(torch.float16).detach().numpy()    
        
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "Conv2d"
    return param_dict

def deconv_hm_fp_get_hardware_params(self, param_dict):
    hmfp_dict = get_hmfp_dict()
    hmfp_conv_dict = hmfp_dict.get("conv")
    act = hmfp_conv_dict.get("act")
    we_dict = hmfp_conv_dict.get("weight")
    hardware_align = hmfp_dict.get('hardware_align', False)
    mode = hmfp_dict.get('mode', 'sefp').lower()
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')
    extra_gpu = hmfp_dict.get('extra_gpu', False)

    ####
    op_class = getattr(self, 'op_class', 'CIMDTransposeConv2d')
    if hasattr(self, 's') and op_class == 'CIMDTransposeConv2d':
        stride = self.s
    elif hasattr(self, 'ori_stride'):
        stride = self.ori_stride
    elif hasattr(self, 'stride'):
        stride = self.stride
    else:
        raise RuntimeError("no stride")
    if isinstance(stride, tuple) and len(stride) == 2:
        assert stride[0] == stride[1], "invalid stride"
        stride = stride[0]
    
    if hasattr(self, 'ori_padding') and op_class == 'ConvTransposed2d':
        padding = self.ori_padding
    elif hasattr(self, 'padding'):
        padding = self.padding
    else:
        raise RuntimeError("no padding")
    if isinstance(padding, tuple) and len(padding) == 2:
        assert padding[0] == padding[1], "invalid padding"
        padding = padding[0]
    
    if hasattr(self, 'raw_weight'):
        weight = self.raw_weight
    elif hasattr(self, 'weight'):
        weight = self.weight
    else:
        raise RuntimeError("no weight")
    
    if hasattr(self, 'raw_bias') and op_class == 'CIMDTransposeConv2d':
        bias = self.raw_bias
    elif hasattr(self, 'bias'):
        bias = self.bias
    else:
        raise RuntimeError("no bias")
    ####
    
    nshare = act.get("nshare", 64)
    assert nshare == we_dict.get("nshare", 64), "two input tensors should have same nshare"
    
    if hasattr(self, 'conv_weight') and self.conv_weight is not None:
        conv_weight = self.conv_weight
    else:
        conv_weight = weight.transpose(0, 1)
        conv_weight = torch.rot90(conv_weight, 2, [2,3])
        conv_weight = conv_weight.reshape(conv_weight.size(0), -1)
        conv_weight = conv_weight.transpose(0, 1)
        setattr(self, 'conv_weight', conv_weight)

    assert we_dict.get("type") == "keep_fp16" or mode not in ['ssfp'], "mode not support yet"
    if hasattr(self, '_quant_weight_val_') and self._quant_weight_val_ is not None:
        val = self._quant_weight_val_
    else:
        val = quant_weight(act, we_dict, self, hardware_align, extra_gpu, mode, dim=0, w=conv_weight)
        setattr(self, '_quant_weight_val_', val)
    if isinstance(val, tuple):
        w_int, w_scale, weight_bit = val
    else:
        conv_weight = val

    assert len(weight.shape) == 4 and weight.shape[-1] == weight.shape[-2], 'invalid kernel_size'
    kernel_size = weight.shape[-1]
    
    ### param_dict
    param_dict["dilations"] = self.dilation
    param_dict["group"] = self.groups
    param_dict["kernel_shape"] = kernel_size
    param_dict["pads"] = padding
    param_dict["strides"] = stride

    param_dict['hmfp'] = mode

    config = act.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    mas = int(config[1])
    rounding = act.get("rounding", 'rne')
    keep_bit_for_exp = act.get("keep_bit_for_exp", True)    
    param_dict['hmfp_act_mas_bit'] = mas + 1
    param_dict['hmfp_act_exp_bit'] = exp
    param_dict['hmfp_act_rounding'] = rounding
    param_dict['hmfp_act_nshare'] = nshare
    param_dict['hmfp_act_keep_bit_for_exp'] = keep_bit_for_exp
    
    config = we_dict.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    rounding = we_dict.get("rounding", 'rne')
    keep_bit_for_exp = we_dict.get("keep_bit_for_exp", True)    
    param_dict['hmfp_weight_mas_bit'] = weight_bit
    param_dict['hmfp_weight_exp_bit'] = exp
    param_dict['hmfp_weight_rounding'] = rounding
    param_dict['hmfp_weight_nshare'] = nshare
    param_dict['hmfp_weight_keep_bit_for_exp'] = keep_bit_for_exp
    
    assert w_int.max() < pow(2, weight_bit-1) and w_int.min() >= -pow(2, weight_bit-1), "error data range"
    if weight_bit > 8:
        hmfp_weight = w_int.cpu().to(torch.int16).detach().numpy()
    else:
        hmfp_weight = w_int.cpu().to(torch.int8).detach().numpy()           
    param_dict["hmfp_weight"] = hmfp_weight
    param_dict["hmfp_weight_scale"] = w_scale.cpu().to(torch.float16).detach().numpy()
    
    param_dict['hmfp_psum_dtype'] = psum_dtype
    param_dict['hmfp_psum_round_mode'] = psum_round_mode
    
    param_dict["have_bias"] = True
    items = ["with_relu"]
    for item in items:
        if item not in param_dict and hasattr(self, item):
            param_dict[item] = getattr(self, item)

    param_dict["weight"] = weight.cpu().to(torch.float16).detach().numpy()
    if self.bias is not None:
        param_dict["bias"] = bias.cpu().to(torch.float16).detach().numpy()
    else:
        param_dict["bias"] = torch.zeros(self.weight.shape[0]).to(torch.float16).detach().numpy()    
        
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "ConvTranspose2d"
    return param_dict

def matmul_hm_fp_get_hardware_params(self, param_dict):
    hmfp_dict = get_hmfp_dict()
    hardware_align = hmfp_dict.get('hardware_align', False)
    hmfp_matmul_dict = hmfp_dict.get("matmul")
    left = hmfp_matmul_dict.get("left")
    right = hmfp_matmul_dict.get("right")
    psum_dtype = hmfp_dict.get('psum_dtype', 'fp32').lower()
    psum_round_mode = hmfp_dict.get('psum_round_mode', 'trunc')

    param_dict["dilations"] = [1, 1]
    param_dict["group"] = 1
    param_dict["kernel_shape"] = [1, 1]
    param_dict["pads"] = [0, 0]
    param_dict["strides"] = [1, 1]

    config = left.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    mas = int(config[1])
    nshare = left.get("nshare", 64)
    rounding = left.get("rounding", 'rne')
    keep_bit_for_exp = left.get("keep_bit_for_exp", True)    
    param_dict['hmfp_act_mas_bit'] = mas + 1
    param_dict['hmfp_act_exp_bit'] = exp
    param_dict['hmfp_act_rounding'] = rounding
    param_dict['hmfp_act_nshare'] = nshare
    param_dict['hmfp_act_keep_bit_for_exp'] = keep_bit_for_exp
    
    config = right.get("type").split("fp16_to_e")[1]
    config = config.split("m")
    exp = int(config[0])
    mas = int(config[1])
    rounding = right.get("rounding", 'rne')
    nshare = left.get("nshare", 64)
    keep_bit_for_exp = right.get("keep_bit_for_exp", True)    
    param_dict['hmfp_weight_mas_bit'] = mas + 1
    param_dict['hmfp_weight_exp_bit'] = exp
    param_dict['hmfp_weight_rounding'] = rounding
    param_dict['hmfp_weight_nshare'] = nshare
    param_dict['hmfp_weight_keep_bit_for_exp'] = keep_bit_for_exp
    
    param_dict['hmfp_psum_dtype'] = psum_dtype
    param_dict['hmfp_psum_round_mode'] = psum_round_mode
   
    param_dict["have_bias"] = 0
    param_dict["with_relu"] = 0
    
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "Matmul"
    return param_dict

def lut_hm_fp_get_hardware_params(self, param_dict):
    param_dict["table_name"] = self.table.func.__name__
    param_dict["table_size"] = self.table.table_size

    param_dict["table_index"] = self.table.index.cpu().to(torch.float16).detach().numpy()
    param_dict["table_value"] = self.table.table.cpu().to(torch.float16).detach().numpy()
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    return param_dict

def softmax_hm_fp_get_hardware_params(self, param_dict):
    axis = self.dim if hasattr(self, 'dim') else 1
    axis = self.axis if hasattr(self, 'axis') else axis
    param_dict["axis"] = axis
    sub_param_dict = self.lut_exp.get_hardware_params()
    for k, v in sub_param_dict.items():
        param_dict['exp_' + k] = v
    sub_param_dict = self.lut_div.get_hardware_params()
    for k, v in sub_param_dict.items():
        param_dict['div_' + k] = v
    
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "Softmax"
    return param_dict

def ln_hm_fp_get_hardware_params(self, param_dict):
    param_dict["eps"] = self.eps.cpu().numpy()
    if param_dict["eps"].size == 1:
        param_dict["eps"] = param_dict["eps"].tolist()
    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "LayerNorm"
    return param_dict

def gn_hm_fp_get_hardware_params(self, param_dict):
    param_dict["eps"] = self.eps.cpu().numpy()
    if param_dict["eps"].size == 1:
        param_dict["eps"] = param_dict["eps"].tolist()

    if hasattr(self, "i_bit"):
        delattr(self, 'i_bit')
    if hasattr(self, "o_bit"):
        delattr(self, 'o_bit')
    if hasattr(self, 'op_class'):
        self.op_class = "GroupNorm"
    return param_dict

def add_hm_fp_get_hardware_params(self, param_dict):
    if hasattr(self, "i_bit"):
        try:
            delattr(self, 'i_bit')
        except:
            pass
    if hasattr(self, "o_bit"):
        try:
            delattr(self, 'o_bit')
        except:
            pass
    if hasattr(self, 'op_class'):
        op_class = self.op_class.lower()
        if 'sub' in op_class:
            self.op_class = "Sub"
        else:
            self.op_class = "Add"
    return param_dict
    
def test_deconv(input, weight, stride=2):
    # input[...] = 0
    # input[0,0,0,0] = 1
    # input[0,0,0,1] = 2
    # input[0,0,1,0] = 3
    # input[0,0,1,1] = 4
    # weight[...] = 0
    # weight[0,0,1,0] = 1
    kernel_size = weight.shape[-1]
    co = weight.shape[1]
    bs, ci, h, w = input.shape
    
    # reference:
    torch_out = F.conv_transpose2d(input, weight, stride=stride)
    
    # step 1. unpool indices
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    grid_h, grid_w = torch.meshgrid(y, x, indexing='ij')
    pool_size = torch.Size([bs, ci, (h-1)*stride + 1, (w-1)*stride + 1])
    indices = grid_w * stride + grid_h * pool_size[-1] * stride

    # step 2. unpool
    indices = indices.to(input.device)
    pool_indices = indices.repeat(bs, ci, 1, 1)
    
    unpool = nn.MaxUnpool2d(stride, stride=stride)
    output = unpool(input, pool_indices, output_size=pool_size)
    
    # step 3. unfold
    unfold = F.unfold(output, (kernel_size, kernel_size), padding=(kernel_size-1, kernel_size-1))
    unfold = unfold.transpose(1, 2)
    
    # step 4. weight
    conv_weight = weight.transpose(0, 1)
    conv_weight = torch.rot90(conv_weight, 2, [2,3])
    conv_weight = conv_weight.reshape(conv_weight.size(0), -1)
    conv_weight = conv_weight.transpose(0, 1)

    # step 4. matmul
    conv_output = torch.matmul(unfold, conv_weight)
    conv_output = conv_output.transpose(1, 2)
    
    # step 5. reshape
    oh = pool_size[-2] + (kernel_size-1) 
    ow = pool_size[-1] + (kernel_size-1)
    out = conv_output.reshape(bs, co, oh, ow)

    print((torch_out - out).abs().max())
    print(torch_out.abs().max())
    
if __name__ == "__main__":    
    input = torch.randn(1, 128, 20, 20)
    weight = torch.randn(128, 128, 2, 2)
    test_deconv(input, weight)