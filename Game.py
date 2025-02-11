import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import random as rd
import math
import time

global_size = 3 if "global_size" not in st.session_state else st.session_state.global_size

def judge(board):
    win_size = 3 if "win_size" not in st.session_state else st.session_state.win_size
    for i in range(global_size - win_size + 1):
        for j in range(global_size):
            check = True
            for k in range(1, win_size):
                if board[global_size * (i + k) + j] != board[global_size * i + j]:
                    check = False
                    break
            if check and board[global_size * i + j] != 0:
                return board[global_size * i + j]
    for i in range(global_size):
        for j in range(global_size - win_size + 1):
            check = True
            for k in range(1, win_size):
                if board[global_size * i + j + k] != board[global_size * i + j]:
                    check = False
                    break
            if check and board[global_size * i + j] != 0:
                return board[global_size * i + j]
    for i in range(global_size - win_size + 1):
        for j in range(global_size - win_size + 1):
            check = True
            for k in range(1, win_size):
                if board[global_size * (i + k) + j + k] != board[global_size * i + j]:
                    check = False
                    break
            if check and board[global_size * i + j] != 0:
                return board[global_size * i + j]
    for i in range(global_size - win_size + 1):
        for j in range(win_size - 1, global_size):
            check = True
            for k in range(1, win_size):
                if board[global_size * (i + k) + j - k] != board[global_size * i + j]:
                    check = False
                    break
            if check and board[global_size * i + j] != 0:
                return board[global_size * i + j]
    for i in range(global_size ** 2):
        if board[i] == 0:
            return 0
    return -1
            

class Node:
    board = [0 for _ in range(global_size ** 2)]
    parent = None
    children = None
    n = 0   # visit times
    w = 0   # value
    x = -1
    y = -1
    expanded = False
    expanded_list = None
    next_num = 2
    def __init__(self, board, parent, x, y):
        self.board = board
        self.parent = parent
        self.next_num = self.board.count(1) - self.board.count(2) + 1    # 1 is O, 2 is X
        self.x = x
        self.y = y
    def make_children(self):
        self.children = []
        if judge(self.board) != 0:
            return
        
        for i in range(global_size ** 2):
            if self.board[i] == 0:
                new_board = self.board.copy()
                new_board[i] = self.next_num
                self.children.append(Node(new_board, self, i // global_size, i % global_size))
        
        self.expanded_list = self.children.copy()

C0 = 1.0
C1 = 1.0
C2 = 0.0
class MCTS:
    start_num = 2

    def __init__(self, start_num=2):
        self.start_num = start_num

    def UCT(self, node: Node, exploration: bool = True):
        # 注意这里的逻辑关系
        return (1 if node.next_num != self.start_num else -1 ) * C0 * node.w / node.n + (C1 * np.sqrt(2.0 * np.log(node.parent.n) / node.n) + C2 * node.x * (global_size - 1 - node.x) * node.y * (global_size - 1 - node.y) / (global_size ** 4)) * exploration

    def selection(self, node: Node, exploration: bool = True):
        if node.children is not None and len(node.children) > 0:
            node = max(node.children, key=lambda child: self.UCT(child, exploration))
        return node

    def expansion(self, node: Node):
        if node.children is None:
            node.make_children()
        
        if len(node.expanded_list) <= 1:
            node.expanded = True
        if len(node.expanded_list) == 0:
            return node
        ret = rd.choice(node.expanded_list)
        node.expanded_list.remove(ret)
        return ret

    def simulation(self, node: Node):
        board = node.board.copy()
        next_num = board.count(1) - board.count(2) + 1
        while judge(board) == 0:
            empty = [i for i in range(global_size ** 2) if board[i] == 0]
            if len(empty) == 0:
                return 0
            board[rd.choice(empty)] = next_num
            next_num = 3 - next_num
        if judge(board) == -1:
            return 0
        return 1 if judge(board) == self.start_num else -1

    def backpropagation(self, node: Node, result: int):
        while node:
            node.n += 1
            node.w += result
            node = node.parent

    def tree_policy(self, node: Node):
        while True:
            if judge(node.board) != 0:
                return node
            if not node.expanded:
                return self.expansion(node)
            node = self.selection(node)

    def search(self, _node: Node):
        start_time = time.time()
        node = _node
        while time.time() - start_time < 1.5 * (global_size ** 2 / 9):
            node = self.tree_policy(_node)
            result = self.simulation(node)
            self.backpropagation(node, result)
        return self.selection(_node, False) # 重要！！！这里exploration要设为False，否则选取的节点非最优！
    
    
def draw_board(able_ai: bool = True):
    for i in range(global_size):
        cols = st.columns(global_size)
        for j in range(global_size):
            index = global_size * i + j
            if st.session_state.board[index] == 0:
                if cols[j].button("_", key=index, disabled=(not st.session_state.able) and able_ai):
                    st.session_state.board[index] = st.session_state.current_player
                    st.session_state.current_player = 3 - st.session_state.current_player
                    st.session_state.able = False
                    st.session_state.last_index = index
                    st.rerun()
            elif st.session_state.board[index] == 1:
                if "last_index" in st.session_state and index == st.session_state.last_index:
                    cols[j].markdown(f"<button style='color: white;'> O </button>", unsafe_allow_html=True)
                else:
                    cols[j].button("O", key=index, disabled=True)
            else:
                if "last_index" in st.session_state and index == st.session_state.last_index:
                    cols[j].markdown(f"<button style='color: white;'> X </button>", unsafe_allow_html=True)
                else:
                    cols[j].button("X", key=index, disabled=True)

def ai_run():
    node = Node(st.session_state.board, None, -1, -1)
    mcts = MCTS(st.session_state.current_player)
    node = mcts.search(node)
    st.session_state.last_index = node.x * global_size + node.y
    st.session_state.board = node.board
    st.session_state.current_player = 3 - st.session_state.current_player
    if st.session_state.hint:
        st.session_state.hint = False
    st.rerun()

if __name__ =="__main__":
    st.set_page_config(
        page_title="Game - Alan_ZZH",
        page_icon=":thinking_face:",
        layout="centered",
        initial_sidebar_state="auto",
        )
    st.title("Game_zzh")
    game_choice=st.selectbox("请选择游戏",["Tic-Tac-Toe", "Sine"])
    if game_choice=="Sine":
        st.header("欢迎来到Sine小游戏！")
        st.write("我们需要确定一个三角函数的表达式，具体形式为y=A*sin(a*x+b)，(0<=A<=2)可是我们并不知道参数是多少，太糟糕了！")
        st.write("请利用我们给出的sin和cos函数，通过调整y=sin(a1*x+b1)和y=cos(a2*x+b2)中a1,a2,b1,b2的值来拟合该函数")
        st.write("要快！我们龙国的科技发展就靠你了！")
        st.write("PS:由于目标函数为标准三角函数，即用于拟合的sin、cos函数的x系数应相同，您可以选择同时调控两个x系数a1、a2，也可以单独调控")
        choice=st.selectbox("请选择是否同时调控a1与a2",["同时调控（默认）","分开调控"])
        if "log" not in st.session_state:   # 记录历史数据（初始值）
            st.session_state["log"]=[
                rd.uniform(0.2,1.8),
                rd.uniform(0.8,4.0),
                rd.uniform(-3.2,3.2),
                1,
                1,
                0,
                0
            ]
        A, a, b,a1,a2,b1,b2=st.session_state["log"] # 从log读取数据
        if choice=="同时调控（默认）":  # 分开展示界面（有点冗余）
            st.write("y=sin(a1 * x + b1)")
            a1=st.slider("(频率)Frequency of Sine and Cosine : a1, a2",min_value=0.0,max_value=5.0,value=1.0,step=0.001)
            a2=a1
            st.session_state["log"][3]=a1
            st.session_state["log"][4]=a2
            b1=st.slider("(sin相位)Phase of Sine : b1",min_value=-5.0,max_value=5.0,value=0.0,step=0.001)
            st.session_state["log"][5]=b1
            st.write("y=cos(a2 * x + b2)")
            b2=st.slider("(cos相位)Phase of Cosine : b2",min_value=-5.0,max_value=5.0,value=0.0,step=0.001)
            st.session_state["log"][6]=b2
        else:
            st.write("y=sin(a1 * x + b1)")
            a1=st.slider("(sin频率)Frequency of Sine : a1",min_value=0.0,max_value=5.0,value=1.0,step=0.001)
            st.session_state["log"][3]=a1
            b1=st.slider("(sin相位)Phase of Sine : b1",min_value=-5.0,max_value=5.0,value=0.0,step=0.001)
            st.session_state["log"][5]=b1
            st.write("y=cos(a2 * x + b2)")
            a2=st.slider("(cos频率)Frequency of Cosine : a2",min_value=0.0,max_value=5.0,value=1.0,step=0.001)
            st.session_state["log"][4]=a2
            b2=st.slider("(cos相位)Phase of Cosine : b2",min_value=-5.0,max_value=5.0,value=0.0,step=0.001)
            st.session_state["log"][6]=b2
        x=np.linspace(0,10,1000)
        y=np.sin(a1*x+b1)+np.cos(a2*x+b2)
        t=A*np.sin(a*x+b)
        fig, ax = plt.subplots()
        ax.plot(x,t,'--',label="Target")
        ax.plot(x,y,'-g',label="Yours")
        ax.set_title('Sine Wave')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0,10)
        ax.set_ylim(-2.0,2.0)
        ax.legend()
        st.pyplot(fig)  # 以上为绘图操作
        col1,col2=st.columns(2) # 建立两个按钮，一个检测是否成功，一个重置
        with col1:
            check=st.button("Check:white_check_mark:")
        with col2:
            reset=st.button("Reset",type="primary")
        if check:
            w=0.0
            for i in range(0,100):
                w+=math.pow(A*math.sin(a*i/10+b)-math.sin(a1*i/10+b1)-math.cos(a2*i/10+b2),2)
            if w < 3:   # 精确性判断。精度为5时自我感觉游玩难度适中偏易，可换为其他数值，目前为3（越小越难）
                st.write("**Success! 你成功了！**")
                st.write("实际函数：y="+str(A)+"sin("+str(a)+"x+("+str(b)+'))')
                st.write("您的函数：y=sin("+str(a1)+"x+("+str(b1)+"))+ cos("+str(a2)+"x+("+str(b2)+'))')
            else:
                st.write("**Fail! 你失败了！**")
        if reset:   # 重置操作：将对话中log（历史记录删除，使得下次重新选择随机数。但a1a2b1b2似乎不随之改变，或许是滑块的性质所致？）
            del st.session_state["log"]
            st.rerun()
    else:
        st.header("欢迎来到Tic-Tac-Toe小游戏！")
        st.write("我们需要确定一个N字棋的胜负，具体形式为n*n的棋盘，O先手，X后手")
        board = [0 for _ in range(global_size ** 2)]
        if "board" not in st.session_state:
            st.session_state["board"] = board
        if "current_player" not in st.session_state:
            st.session_state["current_player"] = 1
        if "able" not in st.session_state:
            st.session_state["able"] = True
        if "global_size" not in st.session_state:
            st.session_state.global_size = 3
        if "win_size" not in st.session_state:
            st.session_state.win_size = 3
        if "hint" not in st.session_state:
            st.session_state.hint = False
        game_size=st.number_input("请输入棋盘大小",min_value=3,max_value=8,value=3,step=1)
        win_size=st.number_input("请输入连成一线的棋子数",min_value=3,max_value=min(game_size, 5),value=3,step=1)
        st.session_state.win_size=win_size
        if global_size != game_size:    # 只有global_size被改变才额外触发rerun，它是全局变量，需要保持一致
            st.session_state.global_size=game_size
            st.session_state["board"] = [0 for _ in range(st.session_state.global_size ** 2)]
            st.rerun()
        game_mode=st.selectbox("请选择对战方式",["玩家与AI","玩家与玩家"])
        if game_mode=="玩家与AI":
            who_first=st.selectbox("请您选择先手或后手",["您先手","您后手"])
        st.write(f"当前玩家: {'O' if st.session_state.current_player == 1 else 'X'}")
        draw_board()
        hint_button=st.button("提示", disabled=st.session_state.hint or not st.session_state.able)
        if hint_button:
            st.session_state.hint = True
            st.session_state.able = False
            st.rerun()
        restart=st.button("重新开始")
        if restart:
            st.session_state.able = True
            st.session_state.board = board
            st.session_state.current_player = 1
            st.rerun()
        if judge(st.session_state.board) != 0:
            st.session_state.able = False
            if judge(st.session_state.board) == -1:
                st.write("平局！")
            elif judge(st.session_state.board) == 1:
                st.write("O 获胜！")
            else:
                st.write("X 获胜！")
        else:
            if game_mode=="玩家与玩家":
                if st.session_state.hint:
                    ai_run()
                elif not st.session_state.able:
                    st.session_state.able = True
                    st.rerun()
            else:
                if ((who_first=="您先手" and st.session_state.current_player == 1) or (who_first=="您后手" and st.session_state.current_player == 2)) and not st.session_state.hint:
                    if not st.session_state.able:
                        st.session_state.able = True
                        st.rerun()
                if ((st.session_state.current_player == 2) if who_first=="您先手" else (st.session_state.current_player == 1)) or st.session_state.hint:
                    if st.session_state.able:
                        st.session_state.able = False
                        st.rerun()
                    ai_run()