#!/usr/bin/env python
from typing import List, Tuple, Dict
import os
import torch
from torch import nn


PACKAGE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.split(PACKAGE)[0], 'data')


def script(cls):
    model = cls().eval()
    model = torch.jit.script(model)
    output_path = os.path.join(DATA, f'{cls.__name__}.pt')
    model.save(output_path)
    print(f"saved model to {output_path}")


class module_that_returns_int(nn.Module):
    def forward(self, _: torch.Tensor) -> int:
        return 222
script(module_that_returns_int)

class module_that_returns_float(nn.Module):
    def forward(self, _: torch.Tensor) -> float:
        return 2.22
script(module_that_returns_float)

class module_that_returns_bool(nn.Module):
    def forward(self, _: torch.Tensor) -> bool:
        return True
script(module_that_returns_bool)

class module_that_returns_string(nn.Module):
    def forward(self, _: torch.Tensor) -> str:
        return "foo"
script(module_that_returns_string)

class module_that_returns_tensor_list(nn.Module):
    def forward(self, _: torch.Tensor) -> List[torch.Tensor]:
        return [torch.zeros(1), torch.ones(2)]
script(module_that_returns_tensor_list)

class module_that_returns_list(nn.Module):
    def forward(self, _: torch.Tensor) -> List[int]:
        return [6, 7, 8]
script(module_that_returns_list)

class module_that_returns_empty_list(nn.Module):
    def forward(self, _: torch.Tensor) -> List[int]:
        return list()
script(module_that_returns_empty_list)

class module_that_returns_tuple(nn.Module):
    def forward(self, _: torch.Tensor) -> Tuple[int, int, int]:
        return (9, 6, 3)
script(module_that_returns_tuple)

class module_that_returns_none(nn.Module):
    def forward(self, _: torch.Tensor) -> None:
        return None
script(module_that_returns_none)

class module_that_returns_dict_float_key(nn.Module):
    def forward(self, _: torch.Tensor) -> Dict[float, str]:
        return {1.23: "foo"}
script(module_that_returns_dict_float_key)

class module_that_returns_dict_int_key(nn.Module):
    def forward(self, _: torch.Tensor) -> Dict[int, str]:
        return {45: "foo"}
script(module_that_returns_dict_int_key)

class module_that_returns_dict_str_key(nn.Module):
    def forward(self, _: torch.Tensor) -> Dict[str, str]:
        return {"bar": "foo"}
script(module_that_returns_dict_str_key)

# class module_that_returns_dict_complex_double_key(nn.Module):
#     def forward(self, _: torch.Tensor) -> Dict[Any, str]:
#         return {}
# script(module_that_returns_dict_complex_double_key)

class module_that_returns_dict_bool_key(nn.Module):
    def forward(self, _: torch.Tensor) -> Dict[bool, str]:
        return {True: "foo"}
script(module_that_returns_dict_bool_key)

# class module_that_returns_dict_tensor_key(nn.Module):
#     def forward(self, _: torch.Tensor) -> Dict[torch.Tensor, str]:
#         return {}
# script(module_that_returns_dict_tensor_key)
