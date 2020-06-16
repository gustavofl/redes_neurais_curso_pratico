import numpy as np
import matplotlib.pyplot as plt

from plot_rn_2d import PlotRN_2D
import util

class Perceptron:

    def __init__(self, qnt_entradas=3, taxa_aprendizado=0.01):
        self.pesos = np.random.rand(qnt_entradas)
        self.epoca = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_normalizacao = None

        self.plot_rn = PlotRN_2D(self)
    
    def feedforward(self, x):
        return x.dot(np.transpose(self.pesos))

    def ativacao(self, u):
        # degrau bipolar
        return (u>=0)*2-1
    
    def salvar(self, caminho):
        dados = {
            'pesos': self.pesos,
            'epoca': self.epoca,
            'taxa_aprendizado': self.taxa_aprendizado,
            'fator_normalizacao': self.fator_normalizacao,
        }

        util.salvar_dados(dados, caminho)

    def carregar(caminho):
        rn = Perceptron()

        dados = util.carregar_dados(caminho)

        rn.pesos = dados['pesos']
        rn.epoca = dados['epoca']
        rn.taxa_aprendizado = dados['taxa_aprendizado']
        rn.fator_normalizacao = dados['fator_normalizacao']

        return rn

    def treino(self, x, d, verbose=False, salvar_imagens=False):
        x = np.copy(x)

        x,self.fator_normalizacao = util.normalizar(x, self.fator_normalizacao)

        possui_erro = True
        while(possui_erro and self.epoca <= 2000):
            possui_erro = False
            
            u = self.feedforward(x)
            y = self.ativacao(u)

            if((y != d).any()):
                self.pesos = self.pesos + self.taxa_aprendizado * (d - y).dot(x)
                possui_erro = True

            if(verbose):
                w0,w1,w2 = self.pesos

                erro = (d - y).dot(x)

                print(self.epoca, erro.sum(), -w1/w2, w0/w2)


            if(salvar_imagens and self.epoca%300 == 0):
                # falta implementar
                
                # self.plot_rn.plotar(x=x, d=d, mostrar=1)
                # self.plot_rn.plotar_classes_aprendidas(imagem_destino='teste.png')
                pass
            
            self.epoca += 1
    
    def classificar(self, x, salvar_imagem=False):
        x = np.copy(x)

        x,fator = util.normalizar(x,self.fator_normalizacao)

        u = self.feedforward(x)
        y = self.ativacao(u)

        return y