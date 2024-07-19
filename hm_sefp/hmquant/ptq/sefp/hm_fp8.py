import torch

def to_share_exp_fp(data:torch.Tensor, dim:int=-1, nshare=32, expbit=8, manbit=6,keep_bit_for_exp=True):
    """
    将nshare个浮点数共享一个指数,返回共享后的tensor
    FP32 has 1 sign, 8 exp, 23 mantissa bits
    """
    raw_shape = data.shape
    raw_dtype = data.dtype
    if dim < 0:
        dim += len(raw_shape)

    # pad
    if (raw_shape[dim] % nshare) != 0:
        padsize = nshare - (raw_shape[dim] % nshare)
        padshape = list(raw_shape)
        padshape[dim] = padsize
        paddata = torch.zeros(padshape, dtype=data.dtype, device=data.device)
        data = torch.cat([data, paddata], dim=dim)
    else:
        padsize = 0

    # 确保输入是连续的float32 tensor
    tensor = data.float().view(torch.int32)
    padded_shape = list(tensor.shape)
    newshape = list(tensor.shape)
    newshape[dim] = newshape[dim] // nshare
    newshape.insert(dim+1, nshare)
    tensor = tensor.reshape(*newshape)

    # 将tensor转换为int32类型
    int_tensor = tensor.view(torch.int32)

    # 提取指数位和mantissa位
    sign_mask = 0x80000000
    exp_mask = 0x7f800000
    mantissa_mask = 0x007FFFFF

    sign_bit = int_tensor & sign_mask
    exp = (int_tensor & exp_mask) >> 23
    mantissa = int_tensor & mantissa_mask

    # 加上前置1
    mantissa[exp != 0] = mantissa[exp != 0] | 0x800000
    mantissa = mantissa >> 1
    
    exp[exp != 0] += 1
    exp[exp == 0] += 2
    # mantissa = mantissa | 0x800000
    
    if expbit<8:
        # FP32 exp本来就只有8
        # clip exp
        exp = (exp-127).clamp(-2**(expbit-1), 2**(expbit-1)-1) + 127

    # 找到最大指数
    max_exp = torch.amax(exp, dim+1, keepdim=True)
    delta_exp = max_exp - exp + (23 - manbit)

    if keep_bit_for_exp:
        if len(delta_exp.shape)==2:
            delta_exp[:,-8:] = delta_exp[:,-8:]+1
        elif len(delta_exp.shape)==4:
            delta_exp[:,:,:,-8:] = delta_exp[:,:,:,-8:]+1
    
    # 截断mantissa
    # rshift = max_exp - exp + (23 - manbit)
    # new_mantissa = (mantissa + (2**(rshift.float() - 1)).int()) >> rshift #四舍五入
    new_mantissa = mantissa >> delta_exp << (23 - manbit) #不做四舍五入
    if len(delta_exp.shape)==2:
        new_mantissa[:,-8:] = new_mantissa[:,-8:] << 1
    elif len(delta_exp.shape)==4:
        new_mantissa[:,:,:,-8:] = new_mantissa[:,:,:,-8:] << 1
    # 缩放结果
    # result = torch.zeros_like(int_tensor, dtype=torch.int32)
    # result = result | sign_bit
    # result = result | (exp << 23)
    # result = result | (new_mantissa & 0x7FFFFF)

    result = new_mantissa.float() / (2 ** 23) * (2 ** (max_exp.float() - 127))
    result = torch.where(sign_bit == 0, result, -result)

    # reshape
    if padsize!=0:
        result = torch.split(result.view(padded_shape), raw_shape[dim], dim=dim)[0]
    result = result.reshape(raw_shape).view(raw_dtype)
    return result

def to_hmfp16(data, dim=-1, nshare=32):
    return to_share_exp_fp(data, dim, nshare, 8, 14)

def to_hmfp8(data, dim=-1, nshare=32):
    return to_share_exp_fp(data, dim, nshare, 8, 6)

# mW4: fp16 => e5m4
def fp16_to_e5m4(data, dim=-1, nshare=256):
    # data clip to fp16 min-max
    data = torch.clamp(data,max=65504,min=-65504)
    data = data.to(torch.float16)
    data = data.float()
    return to_share_exp_fp(data, dim, nshare, 5, 4)

def fp16_to_e5m8(data, dim=-1, nshare=256):
    # data clip to fp16 min-max
    data = torch.clamp(data,max=65504,min=-65504)
    data = data.to(torch.float16)
    data = data.float()
    return to_share_exp_fp(data, dim, nshare, 5, 8)

def fp16_to_e5m10(data, dim=-1, nshare=256):
    # data clip to fp16 min-max
    data = torch.clamp(data,max=65504,min=-65504)
    data = data.to(torch.float16)
    data = data.float()
    return to_share_exp_fp(data, dim, nshare, 5, 10)

def fp16_to_e5m12(data, dim=-1, nshare=256):
    # data clip to fp16 min-max
    data = torch.clamp(data,max=65504,min=-65504)
    data = data.to(torch.float16)
    data = data.float()
    return to_share_exp_fp(data, dim, nshare, 5, 12)

def fp16_to_e8m8(data, dim=-1, nshare=256):
    # data clip to fp16 min-max
    data = torch.clamp(data,max=65504,min=-65504)
    data = data.to(torch.float16)
    data = data.float()
    return to_share_exp_fp(data, dim, nshare, 8, 8)

def keep_fp16(data, dim=-1, nshare=256):
    return data.to(torch.float16)

def to_share_exp_fp16(
    data:torch.Tensor,
    dim=-1,
    nshare=64,
    expbit=5,
    manbit=10,
    rounding = 'rne',
    keep_bit_for_exp=False,
    return_padded=False,
    assemble=True,
    ):
    """
    将nshare个浮点数共享一个指数, 返回共享后的tensor
    FP16 has 1 sign, 5 exp, 10 mantissa bits
    """
    data_shape = data.shape
    data_shape = list(data_shape)
    data_dtype = data.dtype
    assert data_dtype == torch.float16
    assert expbit == 5 and manbit <= 15 and manbit >= 1
    if dim < 0:
        dim += len(data_shape)

    channel = data_shape[dim]
    pad = (nshare - (channel % nshare)) % nshare
    if pad != 0:
        shape = []
        for s in data_shape:
            shape.append(s)
        shape[dim] = pad
        paddata = torch.zeros(shape, dtype=data.dtype, device=data.device)
        tensor = torch.cat([data, paddata], dim=dim)
        paddata = None
    else:
        tensor = data.clone()
    
    tensor_shape = list(tensor.shape)
    shape = []
    for s in tensor_shape:
        shape.append(s)
    shape[dim] = shape[dim] // nshare
    shape.insert(dim + 1, nshare)
    tensor = tensor.reshape(*shape)
    
    # step 1. parse FP16
    # step 1.1
    tensor = tensor.view(torch.int16)
    mask = 0x7C00
    exp = (tensor & mask) >> 10 # ranges from 0 to 31
    exp = exp.to(torch.int8)
    
    is_norm = torch.zeros_like(exp)
    is_norm.masked_fill_(exp != 0, 1)

    mask = 0x03ff
    mantissa = tensor & mask
    mantissa = mantissa.to(torch.float32)
    implicit = torch.zeros_like(tensor)
    implicit.masked_fill_(exp != 0, 1024)
    mantissa.add_(implicit)
    implicit = None
    
    mask = 0x8000
    sign = (tensor & mask) >> (8 + 6)
    sign = sign.to(torch.int8)
    #sign.div(-64)
    sign.add_(1)

    # step 1.2
    exp.sub_(15)
    exp.clamp_min_(-14) # should range from -14 to 15; if exp=16, indicates a NAN or INF number 

    # step 1.3 check
    check = False
    if check:
        tensor = exp.to(torch.float16)
        tensor.sub_(10)  # because mantissa contains 10-bit floating point 
        tensor.exp2_()
        tensor.mul_(sign)
        tensor.mul_(mantissa)
        print("check correctness: ", (tensor.reshape(data.shape) - data).abs().sum())
    tensor = None

    # step 2.1 align norm and sub-norm data
    exp.add_(is_norm)
    shift = is_norm.to(torch.float16)
    shift.exp2_()
    mantissa.div_(shift)
    is_norm = None
    
    # step 2.2 take max exp as the shared exp
    max_exp = torch.amax(exp, dim+1, keepdim=True)
    
    # step 2. search best bias
    bias = 2 # bias should range from 0 to 2
    _min, _max = -pow(2, expbit-1) + bias, pow(2, expbit-1) - 1 + bias
    max_exp.clamp_(min=_min, max=_max)  # ranges from -14 to 15
    
    # step 3. align exp to max_exp:
    shift = max_exp - exp
    exp = None
    shift = shift.to(torch.float16)
    shift.exp2_()
    mantissa.div_(shift)
    shift = None
    
    if check:
        repeat = []
        for i in range(len(sign.shape)):
            repeat.append(1 if i != (dim + 1) else nshare)
        tensor = max_exp.repeat(repeat)
        tensor = tensor.to(torch.float16)
        tensor.sub_(10)  # because mantissa contains 10-bit floating point 
        tensor.exp2_()
        tensor.mul_(sign)
        tensor.mul_(mantissa)
        print("check correctness: ", (tensor.reshape(data.shape) - data).abs().sum())
        tensor = None
    
    # step 4. reserve required mantissa (10 bit implict floating point)
    # step 4.1
    mantissa.mul_(sign)
    sign = None

    # step 4.2
    # reserve MSB-manbit bit
    reseve = torch.zeros(mantissa.shape, dtype=torch.float16, device=mantissa.device)
    reseve.fill_(manbit)
    
    # step 4.3
    # new feature: allocate 1 bit to save exponent 
    if keep_bit_for_exp:
        assert len(reseve.shape) > (dim + 1)
        if dim == 0:
            reseve[:, -8:, ...] -= 1
        elif dim == 1:
            reseve[:, :, -8:, ...] -= 1
        elif dim == 2:
            reseve[:, :, :, -8:, ...] -= 1
        elif dim == 3:
            reseve[:, :, :, :, -8:, ...] -= 1
        elif dim == 4:
            reseve[:, :, :, :, :, -8:, ...] -= 1
        else:
            raise NotImplementedError("dim should not large than 4")

    # step 4.4 remove the least (10 - reseve) bit
    reseve.mul_(-1)
    reseve.add_(10)  # 10 - reseve
    reseve.exp2_()
    
    mantissa.div_(reseve)
    rounding = rounding.lower()
    if rounding == 'rne':
        mantissa.round_()  # RNE by default
    elif rounding == 'rtz':
        mantissa.trunc_()
    elif rounding == 'rup': # up
        mantissa.ceil_()
    elif rounding == 'rdn': # down
        mantissa.floor_()
    else:
        raise RuntimeError("rounding mode {} not supported".format(rounding))
    mantissa.clip_(min=-pow(2, manbit), max=pow(2, manbit) - 1) # clip to range
    if keep_bit_for_exp:
        if dim == 0:
            mantissa[:, -8:, ...].clip_(min=-pow(2, manbit-1), max=pow(2, manbit-1) - 1)
        elif dim == 1:
            mantissa[:, :, -8:, ...].clip_(min=-pow(2, manbit-1), max=pow(2, manbit-1) - 1)
        elif dim == 2:
            mantissa[:, :, :, -8:, ...].clip_(min=-pow(2, manbit-1), max=pow(2, manbit-1) - 1)
        elif dim == 3:
            mantissa[:, :, :, :, -8:, ...].clip_(min=-pow(2, manbit-1), max=pow(2, manbit-1) - 1)
        elif dim == 4:
            mantissa[:, :, :, :, :, -8:, ...].clip_(min=-pow(2, manbit-1), max=pow(2, manbit-1) - 1)
    mantissa.mul_(reseve)
    reseve = None
        
    # step 6. assemble
    if assemble:         
        repeat = []
        for i in range(len(mantissa.shape)):
            repeat.append(1 if i != (dim + 1) else nshare)
        tensor = max_exp.repeat(repeat)
        tensor = tensor.to(torch.float32)
        data_dtype = torch.float32
        tensor.sub_(10) # because mantissa 10-bit floating point
        tensor.exp2_()
        #tensor.mul_(sign)
        tensor.mul_(mantissa)
        max_exp = None
        mantissa = None
    else:
        mantissa.mul_(pow(2, manbit-10))
        max_exp.sub_(manbit)
        mantissa = mantissa.floor_()

    if return_padded and not assemble: # for hardware align
        return mantissa, max_exp.to(torch.float32)
    elif return_padded and assemble:
        raise NotImplementedError
    elif not return_padded and not assemble:
        raise NotImplementedError
    else: # default
        if pad != 0:
            tensor = torch.split(tensor.view(tensor_shape), data_shape[dim], dim=dim)[0]
        tensor = tensor.reshape(data_shape).view(data_dtype)
        return tensor

def to_share_exp_wrapper(tensor, dim=-1, exp=5, mas=10, nshare=64, rounding="rne", keep_bit_for_exp=False,
                         return_padded=False, assemble=True, dtype=None, extra_gpu=False):
    with torch.no_grad():
        if dtype is None:
            dtype = tensor.dtype

        data = tensor.clamp_(max=65504, min=-65504)
        if extra_gpu:
            data = data.to('cuda:1')
        try:            
            data = data.to(torch.float16)
            data = to_share_exp_fp16(data, dim, nshare, exp, mas, rounding, keep_bit_for_exp, return_padded, assemble)
        except Exception as e:
            string = str(e)
            if 'CUDA out of memory' not in string:
                raise e
            data = data.to('cpu')
            data = data.to(torch.float16)
            data = to_share_exp_fp16(data, dim, nshare, exp, mas, rounding, keep_bit_for_exp, return_padded, assemble)

        if assemble:
            if data.dtype != dtype:
                data = data.to(dtype=dtype)
    
            if dtype == tensor.dtype:
                tensor.copy_(data)
            else:
                data = data.to(tensor.device)
                tensor = data
            return tensor
        else:
            return data

        
