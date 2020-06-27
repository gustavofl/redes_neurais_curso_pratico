import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pmc import PMC
import util

def equacao_reta(x, y):
    return (x>0) * (y>0)
    # return y > -x + 5

def ativacao_reta(x, y):
    # degrau bipolar
    return equacao_reta(x,y)*2-1

def gerar_datasets(sufixo):
    util.gerar_dataset_grid('reta_1/dataset_treino_%s.data'%str(sufixo), ativacao_reta, tamanho=400)
    # util.gerar_dataset('reta_1/dataset_treino_%s.data'%str(sufixo), ativacao_reta, tamanho=10)
    util.gerar_dataset('reta_1/dataset_teste_%s.data'%str(sufixo), ativacao_reta, tamanho=1500)

def treinar_rede(dataset, rn=None):
    x,d = dataset

    if(rn == None):
        rn = PMC(topologia=[2,2,2])

    rn.treinar(x, d, verbose=1, guardar_historico=1)

    # rn.plotar_curva_aprendizado('Treino (%d épocas de treinamento)'%rn.epoca)

    rn.plotar_aprendizado(titulo='Treino (%d épocas de treinamento)'%rn.epoca)
    
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

    rn=None
    # rn = PMC.carregar('reta_1/rn_reta_%s.rn'%str(sufixo))

    # rn.plot_rn.plotar_dataset( dataset_treino[0] , dataset_treino[1] )
    
    rn = treinar_rede(dataset_treino, rn=rn)
    # rn.salvar('reta_1/rn_reta_%s.rn'%str(sufixo))

    # testar_rede(rn, dataset_teste)

def teste_pmc():
    rn = PMC(topologia=[2,2])

    x = np.array([[0.4,0.1],[0.1,0.4]])
    d = np.array([[1,-1],[-1,1]])

    rn.treinar(x,d,verbose=1)

if __name__ == "__main__":
    main()
    # teste_pmc()