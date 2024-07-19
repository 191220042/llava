
import torch
import torch.nn as nn
from typing import List

cut_points_dict = {
    "leaky_relu": [-65504., -33824., -25456., -17520., -13032.,  -6656.,  -3914.,  -2015.,
         -2014.,     -0.,  65504.],
    "exp": [-6.5504e+04, -1.7062e+01, -8.3828e+00, -4.5352e+00, -1.1318e+00,
         1.8740e+00,  4.5859e+00,  6.2422e+00,  8.2031e+00,  1.1242e+01,
         6.5504e+04],
    "log": [5.9605e-08, 8.9407e-07, 9.7036e-04, 4.1748e-02, 1.9226e-01, 2.1895e+00,
        1.4211e+01, 7.1125e+01, 9.9000e+02, 4.4352e+04, 6.5504e+04],
    "divide": [5.9605e-08, 1.4663e-05, 3.5095e-04, 4.3068e-03, 1.1359e-01, 1.4189e+00,
        1.0484e+01, 7.0500e+01, 9.7700e+02, 1.5576e+04, 6.5504e+04],
    "silu": [-6.5504e+04, -1.8719e+01, -1.1141e+01, -7.7461e+00, -2.0840e+00,
        -8.2617e-01, -6.8420e-02,  1.4856e-01,  1.1289e+00,  5.3516e+00,
         6.5504e+04],
    "sigmoid": [-6.5504e+04, -1.8484e+01, -8.8750e+00, -6.6641e+00, -5.7109e+00,
        -4.7617e+00, -4.0430e+00,  9.8535e-01,  3.2148e+00,  8.7734e+00,
         6.5504e+04],
    "swish": [-6.5504e+04, -1.1766e+01, -5.7539e+00, -1.2461e+00, -5.4834e-01,
        -7.9590e-02,  1.1444e-02,  1.3538e-01,  1.2451e+00,  3.4922e+00,
         6.5504e+04],
    "inverse_sigmoid": [-9.9951e-01,  9.7156e-06,  1.5092e-04,  7.3719e-04,  5.4893e-03,
         6.8359e-02,  2.9321e-01,  8.2178e-01,  9.5312e-01,  9.9561e-01,
         9.9951e-01],
    "inversesigmoid_1e-05": [-9.9951e-01,  9.7156e-06,  1.5092e-04,  7.3719e-04,  5.4893e-03,
         6.8359e-02,  2.9321e-01,  8.2178e-01,  9.5312e-01,  9.9561e-01,
         9.9951e-01],
    "pow_2.0": [-6.5504e+04, -2.5800e+02, -1.8312e+01, -4.6133e+00, -1.1533e+00,
        -1.4368e-01,  2.4695e-01,  2.2246e+00,  3.4469e+01,  2.5725e+02,
         6.5504e+04],
    "hardswish": [-6.5504e+04, -3.0117e+00, -2.5957e+00, -1.2314e+00, -5.9668e-01,
        -1.6150e-01,  2.4109e-02,  3.3276e-01,  1.4971e+00,  3.0176e+00,
         6.5504e+04],
    "gelu": [-6.5504e+04, -6.0273e+00, -3.9453e+00, -3.0000e+00, -1.2539e+00,
        -1.4355e-01, -2.2369e-02,  8.5449e-02,  6.7334e-01,  3.0117e+00,
         6.5504e+04],
    "tanh": [-6.5504e+04, -4.6641e+00, -2.5703e+00, -1.5195e+00, -6.0205e-01,
        -2.5977e-01,  7.3792e-02,  7.9443e-01,  2.5000e+00,  4.8047e+00,
         6.5504e+04],
    "mish": [-6.5504e+04, -2.1859e+01, -1.2477e+01, -6.5898e+00, -1.9648e+00,
        -7.7051e-01, -8.3008e-02,  3.8422e-02,  9.0771e-01,  3.3340e+00,
         6.5504e+04],
    "elu": [-6.5504e+04, -8.9141e+00, -4.1836e+00, -2.4336e+00, -1.1104e+00,
        -7.7588e-01, -3.0908e-01, -8.6792e-02, -1.3252e-02, -6.7282e-04,
         6.5504e+04],
    "softplus": [-6.5504e+04, -1.7062e+01, -8.3828e+00, -6.3594e+00, -5.3984e+00,
        -3.9043e+00, -3.0254e+00, -1.7422e+00,  1.6182e+00,  4.8203e+00,
         6.5504e+04]
}


class NewTable(nn.Module):
    def __init__(self, func, cut_points: List, table_size=257, min=-65504, max=65504, device="cpu") -> None:
        super().__init__()
        self.func = func
        self.cut_points = cut_points
        self.table_size = table_size
        self.min = min
        self.max = max
        self.num_points = (table_size - 1) // (len(cut_points) + 1)
        self.device = device
        self.creat_table()
        
    def creat_table(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        self.index = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        self.all_points = all_points = [self.min] + self.cut_points + [self.max]
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            x = torch.linspace(start, end, self.num_points + 1).to(self.device)
            
            if i != len(all_points) - 2:
                self.index[i*self.num_points:(i+1)*self.num_points] = x[:-1].to(torch.float16)
                y = self.func(x).clamp(-65504, 65504)
                self.table[i*self.num_points:(i+1)*self.num_points] = y[:-1]
            else:
                self.index[i*self.num_points:] = x.to(torch.float16)
                y = self.func(x).clamp(-65504, 65504)
                self.table[i*self.num_points:] = y
        
    def _forward(self, x, inplace=False):
        # 根据切分点，找到x所在的区间
        indices = torch.bucketize(x, self.index, right=True).clip_(1, self.table_size - 1)
        
        # 计算x到左右两个切分点的距离
        interval = self.index[indices] - self.index[indices - 1]
        interval[interval == 0] = 1e-5
        m1 = (x - self.index[indices - 1] ) / interval
        m2 = 1 - m1
        
        # 计算x的插值
        y = self.table[indices - 1] * m2.half() + self.table[indices] * m1.half()
        return y.half()
    
    def cpu_forward(self, x, inplace=False):
        device = x.device
        x_ = x.to('cpu')
        self.index = self.index.to('cpu')
        self.table = self.table.to('cpu')
        out = self._forward(x_)        
        self.index = self.index.to(device)
        self.table = self.table.to(device)        
        if inplace:
            x.copy_(out)
            return x
        else:
            out = out.to(device)
            return out
    
    def split_forward(self, x, split=4, inplace=False):
        if inplace:
            out = x
        else:
            out = x.clone()
        length = int(len(x) / split)
        split = int(split)
        for i in range(split):
            if i == (split-1):
                local = x[i*length:]
                out[i*length:] = self._forward(local)
            else:
                local = x[i*length: (i+1)*length]
                out[i*length: (i+1)*length] = self._forward(local)
        return out

    def forward(self, x, inplace=False):
        shape = x.shape
        x = x.reshape(-1)
        if torch.numel(x) > pow(2, 28):
            split = 32
        if torch.numel(x) > pow(2, 25):
            split = 8
        else:
            split = 1
        out = self.split_forward(x, split, inplace=inplace)
        out = out.reshape(shape)
        return out
        
    def forward_deprecate(self, x, inplace=False):
        """if inplace=True, re-use buffer of `x`"""
        try:
            if torch.numel(x) > pow(2, 28):
                out = self.split_forward(x, 8, inplace=inplace)
            else:
                out = self._forward(x, inplace=inplace)
        except Exception as e:
            error = str(e)
            if "CUDA out of memory" not in error:
                raise RuntimeError(error)
            try: # if oom, try to split input first
                out = self.split_forward(x, 16, inplace=inplace)
            except Exception as e: # if continue oom, convert to cpu version
                error = str(e)
                if "CUDA out of memory" not in error:
                    raise RuntimeError(error)
                out = self.cpu_forward(x, inplace=inplace)            
        return out
        
    def get_hardware_params(self):
        param_dict = dict()
        param_dict["table_name"] = self.func.__name__
        param_dict["table_size"] = self.table_size

        param_dict["table_index"] = self.index.cpu().to(torch.float16).detach().numpy()
        param_dict["table_value"] = self.table.cpu().to(torch.float16).detach().numpy()
        if hasattr(self, "i_bit"):
            delattr(self, 'i_bit')
        if hasattr(self, "o_bit"):
            delattr(self, 'o_bit')
        return param_dict
    
