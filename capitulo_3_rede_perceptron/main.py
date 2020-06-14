import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
import util

def f(x, y):
    # RETA: y = -x
    return y > -x

def ativacao(u):
    # degrau bipolar
    return u*2-1

def gerar_dataset(arquivo_destino, tamanho=300):
    x = np.random.rand(tamanho,3)*20-10
    x[:,0] = -1

    d = ativacao(f(x[:,1], x[:,2]))

    util.salvar_dados({'x':x,'d':d}, arquivo_destino)

def carregar_dataset(arquivo):
    dados = util.carregar_dados(arquivo)
    x = dados['x']
    d = dados['d']

    return x,d

def treinar_rede(dataset, rn=None):
    x,d = dataset

    if(rn == None):
        while(True):
            rn = Perceptron(qnt_entradas=len(x[0,:]))
            rn.pesos = np.array([0.95,0.52,0.03,0.55])
            print(rn.pesos)
            repetir = input('Criar outra rede neural? (s/n) ')
            if(repetir == 'n'):
                break
    
    pesos_iniciais = rn.pesos

    rn.treino(x, d, verbose=1, salvar_imagens=True)

    print('Pesos iniciais:', pesos_iniciais)
    print('Pesos finais:', rn.pesos)

    print(rn.epoca, 'epocas de treinamento.')

    return rn

def testar_rede(rn, dataset):
    x,d = dataset

    y = rn.classificar(x, salvar_imagem=True)

    """ erros = (y != d).sum()

    erro_perc = erros/float(len(x[0,:])) """

    """ reta_aprendida = rn.construir_reta_aprendida()
    reta_funcao = [[-1,1],[1,-1]]


    util.plotar_dados_2d(
        pontos=x_norm, 
        retas=[reta_aprendida, reta_funcao], 
        imagem_destino='projeto/teste_t1.png') """
    
    plano = rn.construir_plano_aprendido()

    util.plotar_dados_3d(
        pontos=x, 
        planos=[plano], 
        mostrar=1,
        imagem_destino='projeto/teste_t6.png',
        limite=None)

    print()

    for yi in y:
        print(yi)

def main():
    dataset_treino = carregar_dataset('projeto/dataset_treino_projeto.data')
    dataset_teste = carregar_dataset('projeto/dataset_teste_projeto.data')

    rn = Perceptron.carregar('projeto/rn_projeto_t6.rn')
    # rn = treinar_rede(dataset_treino, rn=None)
    # rn.salvar('projeto/rn_projeto_t5.rn')
    testar_rede(rn, dataset_teste)

def auditoria():
    dataset_treino = carregar_dataset('projeto/dataset_treino_projeto.data')
    
    rn = treinar_rede(dataset_treino, rn=None)
    
if __name__ == "__main__":
    main()
    # auditoria()
    # alterar_dataset()