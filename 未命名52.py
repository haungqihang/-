#相互作用
import numpy as np
from scipy.linalg import eig, solve
import cmath
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.linalg import schur
import sympy as sp
from scipy.linalg import schur
import numpy as np
from scipy.linalg import svd, eigvals
import mpmath as mp


# ================== 左边矩阵==================
def construct_coefficient_matrices_left(N, a, b):
    """
    正确构造 H(k) = C + D*z + E/z 的系数矩阵，其中 z = e^{ik}
    返回 C, D, E (均为 4N x 4N)
    """
    tb = 1j * b  # t^b 是纯虚数
    V=0.0
    # ----- A 的分解 -----
    A0 = np.array([
        [V, a, -tb, 0],
        [a, V, a, -tb],
        [tb, a, V, a],
        [0, tb, a, V]
    ], dtype=complex)

    A1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [a,tb, 0, 0]
    ], dtype=complex)

    A_neg1 = np.array([
        [0, 0, -tb, a],
        [0, 0, 0,-tb],
        [0, 0, 0, 0],
        [0,0, 0, 0]
    ], dtype=complex)

    # ----- B 的分解 -----
    B0 = np.array([
        [tb, 0, 0, 0],
        [a, -tb, 0, tb],
        [-tb, 0, tb, a],
        [0,0, 0, -tb]
    ], dtype=complex)

    B1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-tb, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    B_neg1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, tb],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    # B 的厄米共轭相关矩阵
    B0d = B0.conj().T
    B1d = B1.conj().T
    B_neg1d = B_neg1.conj().T

    dim = 4 * N
    C1 = np.zeros((dim, dim), dtype=complex)
    D1 = np.zeros((dim, dim), dtype=complex)
    E_mat1 = np.zeros((dim, dim), dtype=complex)

    # 对角块
    for i in range(N):
        r = slice(4*i, 4*i+4)
        C1[r, r] = A0
        D1[r, r] = A1
        E_mat1[r, r] = A_neg1

    # 次对角块
    for i in range(N-1):
        r1 = slice(4*i, 4*i+4)
        r2 = slice(4*i+4, 4*i+8)

        # 下三角 (i+1, i) : B
        C1[r2, r1] = B0d
        D1[r2, r1] = B_neg1d
        E_mat1[r2, r1] = B1d

        # 上三角 (i, i+1) : B†
        C1[r1, r2] = B0
        # 注意：B† 中 z 的系数是 B_neg1†，z^{-1} 的系数是 B1†
        D1[r1, r2] = B1
        E_mat1[r1, r2] = B_neg1
    tb = -1j * b  # t^b 是纯虚数
    V=0.0
    # ----- A 的分解 -----
    A0 = np.array([
        [V, a, -tb, 0],
        [a, V, a, -tb],
        [tb, a, V, a],
        [0, tb, a, V]
    ], dtype=complex)

    A1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [a,tb, 0, 0]
    ], dtype=complex)

    A_neg1 = np.array([
        [0, 0, -tb, a],
        [0, 0, 0,-tb],
        [0, 0, 0, 0],
        [0,0, 0, 0]
    ], dtype=complex)

    # ----- B 的分解 -----
    B0 = np.array([
        [tb, 0, 0, 0],
        [a, -tb, 0, tb],
        [-tb, 0, tb, a],
        [0,0, 0, -tb]
    ], dtype=complex)

    B1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-tb, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    B_neg1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, tb],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    # B 的厄米共轭相关矩阵
    B0d = B0.conj().T
    B1d = B1.conj().T
    B_neg1d = B_neg1.conj().T

    dim = 4 * N
    C2 = np.zeros((dim, dim), dtype=complex)
    D2 = np.zeros((dim, dim), dtype=complex)
    E_mat2 = np.zeros((dim, dim), dtype=complex)

    # 对角块
    for i in range(N):
        r = slice(4*i, 4*i+4)
        C2[r, r] = A0
        D2[r, r] = A1
        E_mat2[r, r] = A_neg1

    # 次对角块
    for i in range(N-1):
        r1 = slice(4*i, 4*i+4)
        r2 = slice(4*i+4, 4*i+8)

        # 下三角 (i+1, i) : B
        C2[r2, r1] = B0d
        D2[r2, r1] = B_neg1d
        E_mat2[r2, r1] = B1d

        # 上三角 (i, i+1) : B†
        C2[r1, r2] = B0
        # 注意：B† 中 z 的系数是 B_neg1†，z^{-1} 的系数是 B1†
        D2[r1, r2] = B1
        E_mat2[r1, r2] = B_neg1
    matrices = [C1, C2]
    C = block_diag(*matrices)
    matrices = [D1, D2]
    D = block_diag(*matrices)
    matrices = [E_mat1,E_mat2]
    E_mat= block_diag(*matrices)
    
    return C, D, E_mat
# ================== 右边矩阵，加入交错能==================
def construct_coefficient_matrices_right(N, a, b,J):
    """
    正确构造 H(k) = C + D*z + E/z 的系数矩阵，其中 z = e^{ik}
    返回 C, D, E (均为 4N x 4N)
    """
    tb =  b  # t^b 是纯虚数
    
    V=0.0
    # ----- A 的分解 -----
    A0 = np.array([
        [ V, a, tb, 0],
        [a,  V, a, tb],
        [tb, a,  V, a],
        [0, tb, a,  V]
    ], dtype=complex)
    Aj = np.array([
        [J, 0, 0, 0],
        [0, -J, 0, 0],
        [0, 0, J, 0],
        [0,0, 0, -J]
    ], dtype=complex)
    A1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [a,tb, 0, 0]
    ], dtype=complex)

    A_neg1 = np.array([
        [0, 0, tb, a],
        [0, 0, 0,tb],
        [0, 0, 0, 0],
        [0,0, 0, 0]
    ], dtype=complex)

    # ----- B 的分解 -----
    B0 = np.array([
        [tb, 0, 0, 0],
        [a, tb, 0, tb],
        [tb, 0, tb, a],
        [0,0, 0, tb]
    ], dtype=complex)

    B1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    B_neg1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, tb],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    # B 的厄米共轭相关矩阵
    B0d = B0.conj().T
    B1d = B1.conj().T
    B_neg1d = B_neg1.conj().T

    dim = 4 * N
    C1 = np.zeros((dim, dim), dtype=complex)
    D1 = np.zeros((dim, dim), dtype=complex)
    E_mat1 = np.zeros((dim, dim), dtype=complex)

    # 对角块
    for i in range(N):
        r = slice(4*i, 4*i+4)
        C1[r, r] = A0+Aj
        D1[r, r] = A1
        E_mat1[r, r] = A_neg1

    # 次对角块
    for i in range(N-1):
        r1 = slice(4*i, 4*i+4)
        r2 = slice(4*i+4, 4*i+8)

        # 下三角 (i+1, i) : B
        C1[r2, r1] = B0d
        D1[r2, r1] = B_neg1d
        E_mat1[r2, r1] = B1d

        # 上三角 (i, i+1) : B†
        C1[r1, r2] = B0
        # 注意：B† 中 z 的系数是 B_neg1†，z^{-1} 的系数是 B1†
        D1[r1, r2] = B1
        E_mat1[r1, r2] = B_neg1
    tb =  b  # t^b 是纯虚数
    
    V=0.0
    # ----- A 的分解 -----
    A0 = np.array([
        [ V, a, tb, 0],
        [a,  V, a, tb],
        [tb, a,  V, a],
        [0, tb, a,  V]
    ], dtype=complex)
    Aj = np.array([
        [-J, 0, 0, 0],
        [0, J, 0, 0],
        [0, 0, -J, 0],
        [0,0, 0,  J]
    ], dtype=complex)
    A1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [a,tb, 0, 0]
    ], dtype=complex)

    A_neg1 = np.array([
        [0, 0, tb, a],
        [0, 0, 0,tb],
        [0, 0, 0, 0],
        [0,0, 0, 0]
    ], dtype=complex)

    # ----- B 的分解 -----
    B0 = np.array([
        [tb, 0, 0, 0],
        [a, tb, 0, tb],
        [tb, 0, tb, a],
        [0,0, 0, tb]
    ], dtype=complex)

    B1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [tb, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    B_neg1 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, tb],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=complex)

    # B 的厄米共轭相关矩阵
    B0d = B0.conj().T
    B1d = B1.conj().T
    B_neg1d = B_neg1.conj().T

    dim = 4 * N
    C2 = np.zeros((dim, dim), dtype=complex)
    D2 = np.zeros((dim, dim), dtype=complex)
    E_mat2 = np.zeros((dim, dim), dtype=complex)

    # 对角块
    for i in range(N):
        r = slice(4*i, 4*i+4)
        C2[r, r] = A0+Aj
        D2[r, r] = A1
        E_mat2[r, r] = A_neg1

    # 次对角块
    for i in range(N-1):
        r1 = slice(4*i, 4*i+4)
        r2 = slice(4*i+4, 4*i+8)

        # 下三角 (i+1, i) : B
        C2[r2, r1] = B0d
        D2[r2, r1] = B_neg1d
        E_mat2[r2, r1] = B1d

        # 上三角 (i, i+1) : B†
        C2[r1, r2] = B0
        # 注意：B† 中 z 的系数是 B_neg1†，z^{-1} 的系数是 B1†
        D2[r1, r2] = B1
        E_mat2[r1, r2] = B_neg1
    matrices = [C1, C2]
    C = block_diag(*matrices)
    matrices = [D1, D2]
    D = block_diag(*matrices)
    matrices = [E_mat1,E_mat2]
    E_mat= block_diag(*matrices)
    
    return C, D, E_mat


def compute_transmission(E, N, t1a,t1b,t2a,t2b,M,S,J):
    """
    计算透射系数 T_LR(i)
    
    参数：
    Ef_i : float, 费米能级（实部）
    state : int, 每个格点的轨道数
    N : int, 系统长度（格点数）
    H00 : np.ndarray, 形状 (dim, dim), onsite 矩阵
    H01 : np.ndarray, 形状 (dim, dim), 右向跃迁矩阵
    M : int, 器件层数（重复单元数）
    time : int, 迭代次数（默认 25）
    
    返回：
    float, 透射系数实部
    """
    # 复数能量
    time=25
    E = E + 1j * 1e-6
    dim = 4 * N          # 矩阵维度
    I = np.eye(dim*2, dtype=complex)
    I1= np.eye(4*dim, dtype=complex)
    H00_L,H01_L,H10_L=construct_coefficient_matrices_left(N, t1a, t1b)
    H00_R,H01_R,H10_R=construct_coefficient_matrices_right(N, t2a, t2b,M)
    #Hcr=np.kron(np.eye(N),np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    
    Hc1=np.zeros((N,N),dtype=complex)
    Hc1[N-1,N-1]=0.2
    '''
    for i in range(0,N):
        Hc1[i,i]=0.1*i+0.1
    '''
    Hcr=np.kron(Hc1,np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    Hcr=np.block([[Hcr,np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),Hcr]])
    # ---------- 左半无限链表面格林函数 ----------
    ai = H10_L.copy()
    bi = H10_L.conj().T.copy()
    ei = H00_L.copy()
    eg = H00_L.copy()
    
    for _ in range(time):
        mm = np.linalg.inv(E * I - ei)
        eg = eg + ai @ mm @ bi
        ei = ei + ai @ mm @ bi + bi @ mm @ ai
        ai = ai @ mm @ ai
        bi = bi @ mm @ bi
    
    gl = np.linalg.inv(E * I - eg)
    hgh_L = H10_L @ gl @ H10_L.conj().T   # 右自能
    
    # ---------- 右半无限链表面格林函数 ----------
    
    ai = H01_R.copy()
    bi = H01_R.conj().T.copy()
    ei = H00_R.copy()
    eg = H00_R.copy()
    
    for _ in range(time):
        mm = np.linalg.inv(E * I - ei)
        eg = eg + ai @ mm @ bi
        ei = ei + ai @ mm @ bi + bi @ mm @ ai
        ai = ai @ mm @ ai
        bi = bi @ mm @ bi
    
    gr = np.linalg.inv(E * I - eg)
    hgh_R = Hcr @ gr @ Hcr.conj().T   # 左自能
    '''
    # ---------- 器件格林函数递归 ----------
    # 初始块
    GR_ii = np.linalg.inv(E * I - H00_L - hgh_L)
    GR_1j = GR_ii.copy()
    
    for _ in range(2, M + 1):
        GR_ii = np.linalg.inv(E * I - H00_L - H10_L @ GR_ii @ H01_L)
        GR_1j = GR_1j @ H01_L @ GR_ii
    
    # 添加右自能
    GR_ii = np.linalg.inv(np.linalg.inv(GR_ii) - hgh_R)
    GR_1j = GR_1j + GR_1j @ hgh_R @ GR_ii
    '''
    #GR_1j = np.linalg.inv(E*I-H00_L+np.kron(np.array([[0.1,0,0,0,0],[0,0.2,0,0,0],[0,0,0.3,0,0],[0,0,0,0.4,0],[0,0,0,0,0.5]]),np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]))-hgh_L- hgh_R)
    S_t=np.zeros((16*N,16*N))
    S_t[slice(0,4*N),slice(0,4*N)]=S[slice(0,4*N),slice(0,4*N)]
    S_t[slice(4*N,8*N),slice(4*N,8*N)]=-S[slice(0,4*N),slice(0,4*N)]
    S_t[slice(8*N,12*N),slice(8*N,12*N)]=S[slice(4*N,8*N),slice(4*N,8*N)]
    S_t[slice(12*N,16*N),slice(12*N,16*N)]=-S[slice(4*N,8*N),slice(4*N,8*N)]
    GR = np.linalg.inv(E*I1-np.block([[H00_L,H01_L],[H10_L,H00_L]])-np.block([[hgh_L,np.zeros((8*N,8*N))],[np.zeros((8*N,8*N)), hgh_R]])-J*S_t)
    GR_1j=np.block([[GR[slice(0,4*N),slice(8*N,12*N)],np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),GR[slice(4*N,8*N),slice(12*N,16*N)]]])

    # ---------- 透射系数 ----------
    T_R = 1j * (hgh_R - hgh_R.conj().T)   # 右线宽函数
    T_L = 1j * (hgh_L - hgh_L.conj().T)   # 左线宽函数
    # Caroli 公式
    T_LR = np.trace(T_L @ GR_1j @ T_R @ GR_1j.conj().T)
    T_l=np.block([[T_L,np.zeros((8*N,8*N))],[np.zeros((8*N,8*N)),np.zeros((8*N,8*N))]])
    T_r=np.block([[np.zeros((8*N,8*N)),np.zeros((8*N,8*N))],[np.zeros((8*N,8*N)),T_R]])
    #T_LR1=T_LR+np.trace(2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.0@GR@T_r@GR.conj().T)*np.trace(T_l@GR@(2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@GR.conj().T)/np.trace((2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@GR@(T_l+T_r)@GR.conj().T)
    
    return T_LR ,GR,T_l ,T_r
def zixuanjihua(V1,V2, N, t1a,t1b,t2a,t2b,M,S,J):
    SS=np.zeros((8*N,8*N),dtype=complex)
    for i in range(0,1000):
        T,G,T_L,T_R=compute_transmission((V2+4)*i/1000 , N, t1a,t1b,t2a,t2b,M,S,J)
        ni=G@(T_R)@G.conj().T
        si=np.zeros((8,N),dtype=complex)
        for i in range(0,N):
            si[0,i]=ni[4*i,4*i]-ni[4*i+4*N,4*i+4*N]
            si[1,i]=ni[4*i+1,4*i+1]-ni[4*i+4*N+1,4*i+4*N+1]
            si[2,i]=ni[4*i+2,4*i+2]-ni[4*i+4*N+2,4*i+4*N+2]
            si[3,i]=ni[4*i+3,4*i+3]-ni[4*i+4*N+3,4*i+4*N+3]
            si[4,i]=ni[4*i+8*N,4*i+8*N]-ni[4*i+12*N,4*i+12*N]
            si[5,i]=ni[4*i+8*N+1,4*i+8*N+1]-ni[4*i+12*N+1,4*i+12*N+1]
            si[6,i]=ni[4*i+8*N+2,4*i+8*N+2]-ni[4*i+12*N+2,4*i+12*N+2]
            si[7,i]=ni[4*i+8*N+3,4*i+8*N+3]-ni[4*i+12*N+3,4*i+12*N+3]
            SS[4*i  ,4*i  ]+=si[0,i]/1000*(V2+4)
            SS[4*i+1,4*i+1]+=si[1,i]/1000*(V2+4)
            SS[4*i+2,4*i+2]+=si[2,i]/1000*(V2+4)
            SS[4*i+3,4*i+3]+=si[3,i]/1000*(V2+4)
            SS[4*i+4*N,4*i+4*N]+=si[4,i]/1000*(V2+4)
            SS[4*i+4*N+1,4*i+4*N+1]+=si[5,i]/1000*(V2+4)
            SS[4*i+4*N+2,4*i+4*N+2]+=si[6,i]/1000*(V2+4)
            SS[4*i+4*N+3,4*i+4*N+3]+=si[7,i]/1000*(V2+4)
    for i in range(0,1000):
        T,G,T_L,T_R=compute_transmission((V1+4)*i/1000, N, t1a,t1b,t2a,t2b,M,S,J)
        ni=G@(T_L)@G.conj().T
        si=np.zeros((8,N),dtype=complex)
        for i in range(0,N):
            si[0,i]=ni[4*i,4*i]-ni[4*i+4*N,4*i+4*N]
            si[1,i]=ni[4*i+1,4*i+1]-ni[4*i+4*N+1,4*i+4*N+1]
            si[2,i]=ni[4*i+2,4*i+2]-ni[4*i+4*N+2,4*i+4*N+2]
            si[3,i]=ni[4*i+3,4*i+3]-ni[4*i+4*N+3,4*i+4*N+3]
            si[4,i]=ni[4*i+8*N,4*i+8*N]-ni[4*i+12*N,4*i+12*N]
            si[5,i]=ni[4*i+8*N+1,4*i+8*N+1]-ni[4*i+12*N+1,4*i+12*N+1]
            si[6,i]=ni[4*i+8*N+2,4*i+8*N+2]-ni[4*i+12*N+2,4*i+12*N+2]
            si[7,i]=ni[4*i+8*N+3,4*i+8*N+3]-ni[4*i+12*N+3,4*i+12*N+3]
            SS[4*i  ,4*i  ]+=si[0,i]/1000*(V1+4)
            SS[4*i+1,4*i+1]+=si[1,i]/1000*(V1+4)
            SS[4*i+2,4*i+2]+=si[2,i]/1000*(V1+4)
            SS[4*i+3,4*i+3]+=si[3,i]/1000*(V1+4)
            SS[4*i+4*N,4*i+4*N]+=si[4,i]/1000*(V1+4)
            SS[4*i+4*N+1,4*i+4*N+1]+=si[5,i]/1000*(V1+4)
            SS[4*i+4*N+2,4*i+4*N+2]+=si[6,i]/1000*(V1+4)
            SS[4*i+4*N+3,4*i+4*N+3]+=si[7,i]/1000*(V1+4)
    
    return SS  
if __name__ == "__main__" :
    J=0.1
    N =15
    t1a = 1
    t1b = 0.1
    E= 0.2
    V1=0.0
    V2=0.0                                                                                                         
    t2a = 1
    t2b = 0.1 
    M=0.5
    S_0=np.zeros((8*N,8*N))
    #T,G,T_L,T_R=compute_transmission(0.3, N, t1a,t1b,t2a,t2b,M,S_0,J)
    #print(T)
    
    S_new=zixuanjihua(V1,V2, N, t1a,t1b,t2a,t2b,M,S_0,J)
    print(np.trace(S_new))
    
    while np.abs(np.trace(S_new-S_0))>0.001:
        
        print(np.abs(np.trace(S_new-S_0)))
        S_0=S_new
        S_new=0.5*zixuanjihua(V1,V2, N, t1a,t1b,t2a,t2b,M,S_new,J)+0.5*S_new
    
    print(np.trace(S_new))
    '''
    for i in range(0,20):
        S_new=zixuanjihua(V1,V2, N, t1a,t1b,t2a,t2b,M,S_0,0.05*i)
        print(np.trace(S_new))
        
        while np.abs(np.trace(S_new-S_0))>0.001:
            
            #print(np.abs(np.trace(S_new-S_0)))
            S_0=S_new
            S_new=0.5*zixuanjihua(V1,V2, N, t1a,t1b,t2a,t2b,M,S_new,0.05*i)+0.5*S_new
        
        print(np.trace(S_new))
    '''
    
    ''' 
    T=0
    for i in range(0,100):
        Ti,G,T_L,T_R=compute_transmission(V2+(V1-V2)*i/100, N, t1a,t1b,t2a,t2b,M,S_new,J)
        T+=Ti/100
    print(T)
    '''
    '''
    T,G,T_L,T_R=compute_transmission(E, N, t1a,t1b,t2a,t2b,M)
    print(T)
    
    T1,G1,T_L1,T_R1=compute_transmission(E, N, t1a,-t1b,t2a,t2b,M)
    T_L2=T_L1-T_L.conj()
    G2=G-G1.T
    
    F=np.zeros((8,N),dtype=complex)
    F1=np.zeros((2,N),dtype=complex)
    F2=np.zeros((2,N),dtype=complex)
    o=G@(T_L)@G.conj().T-G1@(T_L1)@G1.conj().T

    #o=1j*G@(T_L+2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5*np.trace(T_L@G@(2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G.conj().T)/np.trace((2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G@(T_L+T_R)@G.conj().T))@G.conj().T - 1j*G1@(T_L1+2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5*np.trace(T_L1@G1@(2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G1.conj().T)/np.trace((2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G1@(T_L1+T_R1)@G1.conj().T))@G1.conj().T
    #o=1j*G@(T_L+2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.0)@G.conj().T - 1j*G1@(T_L1+2*np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.0)@G1.conj().T
    
    for i in range(0,N):
        F[0,i]=o[4*i,4*i]
        F[1,i]=o[4*i+1,4*i+1]
        F[2,i]=o[4*i+2,4*i+2]
        F[3,i]=o[4*i+3,4*i+3]
        F[4,i]=o[4*i+4*N,4*i+4*N]
        F[5,i]=o[4*i+4*N+1,4*i+4*N+1]
        F[6,i]=o[4*i+4*N+2,4*i+4*N+2]
        F[7,i]=o[4*i+4*N+3,4*i+4*N+3]
        F2[1,i]=o[4*i,4*i]+o[4*i+1,4*i+1]+o[4*i+2,4*i+2]+o[4*i+3,4*i+3]
        F2[0,i]=o[4*i+4*N,4*i+4*N]+o[4*i+4*N+1,4*i+4*N+1]+o[4*i+4*N+2,4*i+4*N+2]+o[4*i+4*N+3,4*i+4*N+3]
        F1[1,i]=G[4*i,4*i]+G[4*i+1,4*i+1]+G[4*i+2,4*i+2]+G[4*i+3,4*i+3]
        F1[0,i]=G[4*i+4*N,4*i+4*N]+G[4*i+4*N+1,4*i+4*N+1]+G[4*i+4*N+2,4*i+4*N+2]+G[4*i+4*N+3,4*i+4*N+3]
    Vw1=-F1.imag/np.pi
    plt.imshow(F.real,cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-5, vmax=5)
    plt.colorbar(label='Value')
    '''
    '''
    X=0
    for i in range(0,8*N):
        X+=-o1[i,i].imag/np.pi
    '''
    #print(np.trace(T_L@G@(-np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G.conj().T)/np.trace((-np.block([[np.kron(np.eye(N),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])),np.zeros((4*N,4*N))],[np.zeros((4*N,4*N)),np.zeros((4*N,4*N))]])*0.5)@G@(T_L+T_R)@G.conj().T))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
