# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:16:13 2023

@author: yamay
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
sys.path.append("../")
sys.path.append("../../../")
sys.path.append("../")
sys.path.append("../../../")
from board_manager import Shape
from board_manager import BOARD_DATA
from deepqnet import DeepQNetwork
import torch
import random

tutorial_data = np.load("./test_data10.npz") #1~10までのサンプル用状態データをロード

colors = [[255,255,255],
          [255,0,0],
          [0,255,0],
          [255,255,0], 
          [218, 179, 0],
          [247, 171, 166],
          [0,0,255],
          [255,255,0]]

#ブロック可視化関数 (numpy 配列からテトリスの盤面を可視化)
def visualize_block(data, shape, size=0.5):
    block_array = []
    fig, ax = plt.subplots()
    fig.set_figwidth(shape[0] * size)
    fig.set_figheight(shape[1] * size)
    for i in range(shape[0]):
        row = []
        for j in range(shape[1]):
            c = colors[int(data[i][j])]
            row.append(c)
        block_array.append(row)
    block_array = np.array(block_array)
    im = plt.imshow(block_array)

    #グリッド線を引く
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 23, 1.0), minor=True)
    ax.set_xticks(np.arange(-0.5, 10,10))
    ax.set_yticks(np.arange(-0.5, 23, 3))
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    ax.imshow(block_array)
    
shape = tutorial_data["1"].shape 
visualize_block(tutorial_data["1"],shape) #サンプル用状態データ１を可視化

# 2値関数
def get_reshape_backboard(board):
    reshape_board = np.where(board > 0, 1, 0)
    return reshape_board

def getShapeCoordArray(Shape_class, direction, x, y):
    coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
    return coordArray

# ブロックを配置
def dropDownWithDy(board, Shape_class, direction, x, dy, board_data_width=10):
    _board = board
    coordArray = getShapeCoordArray(Shape_class, direction, x, 0)
    for _x, _y in coordArray:
        _board[(_y + dy) * board_data_width + _x] = Shape_class.shape
    return _board

def dropDown(board, Shape_class, direction, x,board_data_height=22, board_data_width=10, ShapeNone_index=0):
    dy = board_data_height - 1
    coordArray = getShapeCoordArray(Shape_class, direction, x, 0)
    # update dy
    for _x, _y in coordArray:
        _yy = 0
        while _yy + _y < board_data_height and (_yy + _y < 0 or board[(_y + _yy) * board_data_width + _x] == ShapeNone_index):
            _yy += 1
        _yy -= 1
        if _yy < dy:
            dy = _yy
    _board = dropDownWithDy(board, Shape_class, direction, x, dy)
    return _board


# ブロックを配置後の盤面を取得
def getBoard(board_backboard, Shape_class, direction, x):
    board = copy.deepcopy(board_backboard)
    _board = dropDown(board, Shape_class, direction, x)
    return _board


# ブロックを配置できる範囲を取得
def getSearchXRange(Shape_class, direction,board_data_width=10):
    minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
    xMin = -1 * minX
    xMax = board_data_width - maxX
    return xMin, xMax

# 盤面を2値化する関数
def get_reshape_backboard(board,height=22, width=10):
    board = np.array(board)
    reshape_board = board.reshape(height,width)
    reshape_board = np.where(reshape_board > 0, 1, 0)
    return reshape_board

#次の状態を取得
def get_next_states_v2(curr_backboard,piece_id,CurrentShape_class):
    states = {}
    if piece_id == 5: 
        num_rotations = 1
    elif piece_id == 1 or piece_id == 6 or piece_id == 7:
        num_rotations = 2
    else:
        num_rotations = 4
        
    for direction0 in range(num_rotations):
        x0Min, x0Max = getSearchXRange(CurrentShape_class, direction0)
        for x0 in range(x0Min, x0Max):
            board = getBoard(curr_backboard, CurrentShape_class, direction0, x0)
            reshape_backboard = get_reshape_backboard(board)
            reshape_backboard = torch.from_numpy(reshape_backboard[np.newaxis, :, :]).float()
            states[(x0, direction0)] = reshape_backboard
    return states

model = torch.load("../../../../weight/DQN/sample_weight.pt") #モデルをロード

# ###index を1~10の番号に変えてみてください###

data_index =  1

# #######################################

tutorial_data = np.load("./test_data10.npz") #1~10までのサンプル用状態データをロード
load_data = tutorial_data[repr(data_index)] 
shape = load_data.shape
visualize_block(load_data ,shape) #データを可視化

state = get_reshape_backboard(load_data) # 1,0 の2値データに変更
print(state)
# states =np.array([state]) # 3Dに変更　(batch_size,Height,Width) 今回はbatch_size=1とする
states =np.array([[state]]) # 3Dに変更　(batch_size,Height,Width) 今回はbatch_size=1とする
next_states = torch.from_numpy(states.astype(np.float32)).clone() #numpy -> tensor 型に変更

model.eval()
# print(next_states)
with torch.no_grad():
    predictions = model(next_states) #状態の価値を推論　(modelのforward()が実行される)
print(predictions) #状態の価値（大きい方が価値が高い）

