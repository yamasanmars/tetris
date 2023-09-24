# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:03:04 2023

@author: yamay
"""

import torch

a = torch.arange(6).reshape(2, 3)
b = torch.arange(6).reshape(2, 3)
print(a)
print(a.size())
print()

c = torch.stack([a, b], dim=1) # 新しいdimで連結する. 既存のdimのsizeはすべて揃っている必要がある (numpy.stack)
print("torch.stackで新しい次元（dim2）方向に連結:")
print(c)
print(c.size())
print(c[1][0][0])

print(torch.argmax(c))

# 出力
# torch.stackで新しい次元（dim2）方向に連結:
# tensor([[[0, 0],
#          [1, 1],
#          [2, 2]],
# 
#         [[3, 3],
#          [4, 4],
#          [5, 5]]])
# torch.Size([2, 3, 2])