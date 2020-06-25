import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

def normalizar(array_numpy, fator):
    if(fator == None):
        fator = maior_valor_absoluto(array_numpy)

    array_numpy /= fator

    return array_numpy,fator

def maior_valor_absoluto(array_numpy):
    return max(abs(array_numpy.min()), abs(array_numpy.max()))

def salvar_dados(dados, caminho):
    arq = open(caminho, 'wb')
    pickle.dump(dados, arq)
    arq.close()

def carregar_dados(caminho):
    arq = open(caminho, 'rb')
    dados = pickle.load(arq)
    arq.close()
    
    return dados

def txt_para_dataset(arquivo_origem, arquivo_destino):
    x = []
    d = []

    while(True):
        try:
            dados = [float(i) for i in input().split(' ')]
        except EOFError:
            break

        x.append(dados[:-1])
        d.append(dados[-1])
    
    x = np.array(x)
    d = np.array(d)

    util.salvar_dados({'x':x,'d':d}, arquivo_destino)

def gerar_dataset(arquivo_destino, func_ativacao, tamanho=300):
    x = np.random.rand(tamanho,2)*20-10

    d = func_ativacao(x[:,0], x[:,1])

    salvar_dados({'x':x,'d':d}, arquivo_destino)

def gerar_dataset_grid(arquivo_destino, func_ativacao, tamanho=400):
    r = np.arange(-10, 10+1e-9, 20/(math.sqrt(tamanho)-1))
    xx,yy = np.meshgrid( r , r )

    x = np.empty([xx.size,2])
    x[:,0] = xx.flatten()
    x[:,1] = yy.flatten()

    d = np.empty([xx.size,2])
    d[:,0] = -func_ativacao(x[:,0], x[:,1])
    d[:,1] = func_ativacao(x[:,0], x[:,1])

    salvar_dados({'x':x,'d':d}, arquivo_destino)

def carregar_dataset(arquivo):
    dados = carregar_dados(arquivo)
    x = dados['x']
    d = dados['d']

    return x,d
