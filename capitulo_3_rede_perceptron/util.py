import pickle
import numpy as np
import matplotlib.pyplot as plt

def normalizar(array_numpy, fator):
    if(fator == None):
        fator = maior_valor_absoluto(array_numpy)

    array_numpy[:,1:] /= fator

    return array_numpy,fator

def maior_valor_absoluto(array_numpy):
    return max(abs(array_numpy[:,1:].min()), abs(array_numpy[:,1:].max()))

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

def gerar_dataset(arquivo_destino, func_ativacao, tamanho=300, valor_maximo=10):
    x = np.random.rand(tamanho,3)*2*valor_maximo-valor_maximo
    x[:,0] = -1

    d = func_ativacao(x[:,1], x[:,2])

    salvar_dados({'x':x,'d':d}, arquivo_destino)

def carregar_dataset(arquivo):
    dados = carregar_dados(arquivo)
    x = dados['x']
    d = dados['d']

    return x,d
