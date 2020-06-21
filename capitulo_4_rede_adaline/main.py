import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from adaline import Adaline
import util

def equacao_reta(x, y):
    # RETA: y = 0.25*x + 0.5
    return y > -1*x + 5

def ativacao_reta(x, y):
    # degrau bipolar
    return equacao_reta(x,y)*2-1

def gerar_datasets(sufixo):
    util.gerar_dataset('reta_1/dataset_treino_%s.data'%str(sufixo), ativacao_reta, tamanho=300)
    util.gerar_dataset('reta_1/dataset_teste_%s.data'%str(sufixo), ativacao_reta, tamanho=1500)

def treinar_rede(dataset, rn=None):
    x,d = dataset

    if(rn == None):
        rn = Adaline(qnt_entradas=len(x[0,:]))

    rn.treino(x, d, verbose=1, guardar_historico=1)

    # rn.plotar_animacao(x,d,titulo='Treino (%d épocas de treinamento)'%rn.epoca)
    
    # rn.salvar_animacao(x,d,titulo='Treino (%d épocas de treinamento)'%rn.epoca,nome_arquivo='reta_1/animacao.mp4')

    return rn

def testar_rede(rn, dataset):
    x,d = dataset
    
    y = rn.classificar(x)

    erro = (y != d).sum()
    erro_perc = 100*erro/float(d.size)

    rn.plotar_aprendizado(x,d,titulo='Teste (%.4f%% de erro)'%erro_perc)

def main():
    sufixo = '1'

    gerar_datasets(sufixo)

    dataset_treino = util.carregar_dataset('reta_1/dataset_treino_%s.data'%str(sufixo))
    dataset_teste = util.carregar_dataset('reta_1/dataset_teste_%s.data'%str(sufixo))

    rn = Adaline.carregar('reta_1/rn_reta_%s.rn'%str(sufixo))
    
    rn = treinar_rede(dataset_treino, rn=rn)
    # rn.salvar('reta_1/rn_reta_%s.rn'%str(sufixo))

    # testar_rede(rn, dataset_treino)

if __name__ == "__main__":
    main()