import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from perceptron import Perceptron
import util

def equacao_reta(x, y):
    # RETA: y = 0.25*x + 0.5
    return y > -1*x + 5

def ativacao_reta(x, y):
    # degrau bipolar
    return equacao_reta(x,y)*2-1

def gerar_datasets(sufixo):
    util.gerar_dataset('reta_2/dataset_treino_%s.data'%str(sufixo), ativacao_reta, tamanho=300)
    util.gerar_dataset('reta_2/dataset_teste_%s.data'%str(sufixo), ativacao_reta, tamanho=1500)

def treinar_rede(dataset, rn=None):
    x,d = dataset

    if(rn == None):
        rn = Perceptron(qnt_entradas=len(x[0,:]))

    rn.treino(x, d, verbose=1, guardar_historico=1)

    rn.plotar_aprendizado_animacao(x,d,titulo='Treino (%d épocas de treinamento)'%rn.epoca)

    return rn

def testar_rede(rn, dataset):
    x,d = dataset
    
    y = rn.classificar(x)

    erro = (y != d).sum()
    erro_perc = 100*erro/float(d.size)

    rn.plotar_aprendizado(x,d,titulo='Teste (%.4f%% de erro)'%erro_perc)

def main():
    sufixo = '1'

    # gerar_datasets(sufixo)

    dataset_treino = util.carregar_dataset('reta_2/dataset_treino_%s.data'%str(sufixo))
    dataset_teste = util.carregar_dataset('reta_2/dataset_teste_%s.data'%str(sufixo))

    # rn = Perceptron.carregar('reta_2/rn_reta_3.rn')
    
    rn = treinar_rede(dataset_treino, rn=None)
    rn.salvar('reta_2/rn_reta_%s.rn'%str(sufixo))

    testar_rede(rn, dataset_teste)

if __name__ == "__main__":
    main()