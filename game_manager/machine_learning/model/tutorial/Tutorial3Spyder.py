# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:33:33 2023

@author: yamay
"""

import matplotlib.pyplot as plt

import numpy as np
import copy
import sys
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
          [0, 0, 255],
          [255, 255, 0],
          [0, 0, 0]         
         ]


#サンプルしたブロックを可視化する関数 
def visualize_sample_block(index, size=0.5):
    block_array = []
    fig, ax = plt.subplots()
    fig.set_figwidth(4 * size)
    fig.set_figheight(4 * size)
    if index==1:
        data = np.array([[0, 1, 0, 0],[0,1, 0, 0],[0, 1, 0, 0], [0, 1, 0, 0]])
    elif index==2:
        data = np.array([[0, 0, 0, 0], [0, 1, 0, 0],[0, 1, 0, 0], [0, 1, 1, 0]])
    elif index==3:
        data = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0]])
    elif index==4:
        data = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]])
    elif index==5:
        data = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]])
    elif index==6:
        data = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]])
    elif index==7:
        data = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

    for i in range(4):
        row = []
        for j in range(4):
            c = colors[index*data[i][j]]
            row.append(c)
        block_array.append(row)
    block_array = np.array(block_array)
    im = plt.imshow(block_array)
    
    #グリッド線を引く
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1.0), minor=True)
    ax.set_xticks(np.arange(-0.5, 4,10))
    ax.set_yticks(np.arange(-0.5, 4, 3))
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    ax.imshow(block_array)


#ブロック可視化関数 (numpy 配列からテトリスの盤面を可視化)
def visualize_block(data_list ,shape, size=0.5, title=None):
    fig, ax = plt.subplots(1,len(data_list))
    if not isinstance(title , list):
        title = [None for i in range(len(data_list))]
    
    for k in range(len(data_list)):
        data = data_list[k]
        block_array = []
        
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
        
        if len(data_list)==1:
            #グリッド線を引く
            ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 23, 1.0), minor=True)
            ax.set_xticks(np.arange(-0.5, 10,10))
            ax.set_yticks(np.arange(-0.5, 23, 3))
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
            ax.imshow(block_array)
        
        else:
            #グリッド線を引く
            ax[k].set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax[k].set_yticks(np.arange(-0.5, 23, 1.0), minor=True)
            ax[k].set_xticks(np.arange(-0.5, 10,10))
            ax[k].set_yticks(np.arange(-0.5, 23, 3))
            ax[k].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            ax[k].grid(which='major', color='black', linestyle='-', linewidth=0.5)
            ax[k].imshow(block_array)
        
        if title[k] is not None:
            ax[k].set_title(title[k])
    
shape = tutorial_data["1"].shape 

#サンプル用状態データ１を可視化
visualize_block([tutorial_data["1"], tutorial_data["2"], tutorial_data["3"]], shape, title=["1", "2", "3"])
visualize_block([tutorial_data["4"], tutorial_data["5"], tutorial_data["6"]], shape, title=["4", "5", "6"]) 
visualize_block([tutorial_data["7"], tutorial_data["8"], tutorial_data["9"]], shape, title=["7", "8", "9"]) 

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

# ###index を1~10の番号に変更してください###

data_index =  1 

# #######################################
tutorial_data = np.load("./test_data10.npz") #1~10までのサンプル用状態データをロード
state = tutorial_data[repr(data_index)] 
shape = state.shape
visualize_block([state] ,shape) #データを可視化

BOARD_DATA.clear()
BOARD_DATA.createNewPiece()
ShapeClass, ShapeIdx, ShapeRange = BOARD_DATA.getShapeData(1) #ブロックをサンプリング
curr_backboard = state.flatten() #2次元の配列を1次元のリストに変換
next_steps = get_next_states_v2(curr_backboard, ShapeIdx, ShapeClass) #次の状態を取得
key = random.choice(list(next_steps.keys()))

next_states = next_steps[key] #テンソルをnumpy配列に変換
next_state = next_states[0]
shape = next_state.shape

visualize_sample_block(ShapeIdx) #サンプルしたブロック
visualize_block([state, next_state] ,shape, title=["state", "next_state"]) #データを可視化

state = next_state

model = torch.load("../../../../weight/DQN/sample_weight.pt") #モデルをロード

# ###index を1~10の番号に変更してください###

data_index =  1 

# #######################################
tutorial_data = np.load("./test_data10.npz") #1~10までのサンプル用状態データをロード
state = tutorial_data[repr(data_index)] 
state = get_reshape_backboard(state)
shape = state.shape
visualize_block([state] ,shape) #データを可視化

BOARD_DATA.clear()
BOARD_DATA.createNewPiece()
ShapeClass, ShapeIdx, ShapeRange = BOARD_DATA.getShapeData(1) #ブロックをサンプリング
curr_backboard = state.flatten() #2次元の配列を1次元のリストに変換
next_steps = get_next_states_v2(curr_backboard, ShapeIdx, ShapeClass) #次の状態を取得

next_actions, next_states = zip(*next_steps.items()) #次の状態を展開
next_states = torch.stack(next_states) #テンソル型に変換

model.eval()
predictions = model(next_states)[:, 0]
index = torch.argmax(predictions).item()
action = next_actions[index]


next_states = next_steps[action] #テンソルをnumpy配列に変換
next_state = next_states[0]
shape = next_state.shape

visualize_sample_block(ShapeIdx) #サンプルしたブロック

def cleared_state(state):
    state = np.array(state)
    width = state.shape[1]
    del_row = []
    for h in range(state.shape[0]):
        if width == np.sum(state[h]):
            del_row.append(h)
    del_row.reverse()
    
    for row in del_row:
        print(row)
        state = np.delete(state, row, 0)
        zero_line = np.array([0 for col in range(state.shape[1])])
        state = np.vstack([zero_line,state])
    return state
        
clear_next_state = cleared_state(next_state)

#visualize_block([state, - next_state + (state * 2) , clear_next_state] ,
#                shape, title=["state", "next_state", "cleared"]) #データを可視化

visualize_block([state, (next_state - state)* 2 + state, clear_next_state] ,
                shape, title=["state", "next_state", "cleared"]) #データを可視化(緑が配置したブロック)
state = clear_next_state

def get_state_with_block_id_weight(state, id):
    id_weight = 1.0/id
    print(id_weight)
    return state * id_weight

get_state_with_block_id_weight(state, 1)