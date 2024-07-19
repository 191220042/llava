
import torch

def check(x, bit_sign=1, bit_exp=8, bit_mas=15):
    assert x.dtype in [torch.int32, torch.float32], "format invalid"
    assert bit_exp == 8, "format invalid"
    assert bit_mas <= 23, "format invalid"

def fp32_to_fp24(x, bit_sign=1, bit_exp=8, bit_mas=15, round_mode='trunc'):
    # round_mode in ['trunc', 'RNE']
    check(x, bit_sign, bit_exp, bit_mas)
    x = x.to(torch.float32)
    x_int = x.view(torch.int32)
    mask = 0xFFFFFF00
    x_int_1 = x_int & mask
    round_mode = round_mode.lower()
    if round_mode == 'trunc':
        x_int = x_int_1
    elif round_mode == 'rne':
        mask = 0x7F800000
        x_int_exp = x_int & mask
        mask = 0x007FFF00
        x_int_mas = x_int & mask
        x_int_mas = x_int_mas + 0x00000100
        x_int_exp += x_int_mas == 0x00800000
        x_int_mas[x_int_mas == 0x00800000] = 0
        mask = 0x80000000
        x_int_up = x_int & mask
        x_int_up = x_int_up | x_int_exp | x_int_mas
        x_int_exp = x_int_mas = None

        mask = 0x000000FF
        x_int_2 = x_int & mask
        round_up = (x_int_2 > 0x80) | ((x_int_2 == 0x80) & (x_int_1 % 2 == 1))
        x_int = torch.where(round_up, x_int_up, x_int_1)
        x_int_up = x_int_1 = x_int_2 = None
    else:
        raise RuntimeError("unkonwn rounding mode {}".format(round_mode))
        
    x = x_int.view(torch.float32)
    return x

def test():
    fp32 = torch.rand(100).to(torch.float32)
    
    fp24 = fp32_to_fp24(fp32, round_mode='rne')
    div = fp24 / fp32
    print(div.min().item(), div.max().item(), )
    print((fp24 - fp32).abs().max().item(), (fp24 - fp32).abs().sum().item())
    
    fp24_2 = fp32_to_fp24(fp32)
    div = fp24_2 / fp32
    print(div.min().item(), div.max().item(), )
    print((fp24_2 - fp32).abs().max().item(), (fp24_2 - fp32).abs().sum().item())
    
    print((fp24_2 - fp24).abs().max().item(), (fp24_2 - fp24).abs().sum().item())
    print("")
    
if __name__ == "__main__":
    test()
