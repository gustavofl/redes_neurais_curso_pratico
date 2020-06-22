import numpy as np
import matplotlib.pyplot as plt

from plot_rn_2d import PlotRN_2D
import util

class Adaline:

    def __init__(self, qnt_entradas=3, taxa_aprendizado=0.001):
        self.pesos = np.random.rand(qnt_entradas)
        self.epoca = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_normalizacao = None

        self.historico_pesos = []
        self.historico_erro = []
        self.plot_rn = PlotRN_2D(self, self.historico_pesos)
    
    def feedforward(self, x):
        return x.dot(np.transpose(self.pesos))

    def ativacao(self, u):
        # degrau bipolar
        return (u>=0)*2-1

    def treino(self, x, d, verbose=False, guardar_historico=False):
        x = np.copy(x)

        x,self.fator_normalizacao = util.normalizar(x, self.fator_normalizacao)

        erro_minimo = 1e-12
        ultimo_erro = 1
        while(self.epoca <= 1000):
            u = self.feedforward(x)

            erro = ((d-u)**2).sum()/float(d.size)

            if(abs(erro-ultimo_erro) <= erro_minimo):
                break

            self.pesos = self.pesos + self.taxa_aprendizado * (d - u).dot(x)

            ultimo_erro = erro
            self.epoca += 1
            
            if(verbose):
                w0,w1,w2 = self.pesos
                # print('%04d - %.12f - [ %.12f , %.12f , %.12f ]'%(self.epoca, erro, -w1/w2, w0/w2))
                # print('%04d - %.12f - [ %.12f , %.12f , %.12f ]'%(self.epoca, erro, w0, w1, w2))
                # print(self.epoca, erro, -w1/w2, w0/w2)
                print('%04d - %.12f'%(self.epoca, erro))

            if(guardar_historico and self.epoca%1 == 0):
                self.historico_pesos.append(np.copy(self.pesos))
                self.historico_erro.append(erro)

    def classificar(self, x, salvar_imagem=False):
        x = np.copy(x)

        x,fator = util.normalizar(x,self.fator_normalizacao)

        u = self.feedforward(x)
        y = self.ativacao(u)

        return y
    
    def plotar_animacao(self, x=None, d=None, titulo=''):
        if(len(self.historico_pesos) > 0):
            self.plot_rn.plotar_animacao(x, d, titulo=titulo)
    
    def salvar_animacao(self, x=None, d=None, titulo='', nome_arquivo=''):
        if(len(self.historico_pesos) > 0):
            self.plot_rn.salvar_animacao(x, d, titulo=titulo, nome_arquivo=nome_arquivo)

    def plotar_aprendizado(self, x=None, d=None, titulo=''):
        self.plot_rn.plotar_aprendizado(x, d, titulo=titulo)

    def plotar_curva_aprendizado(self, titulo=''):
        self.plot_rn.plotar_curva_aprendizado(titulo=titulo)
    
    def salvar(self, caminho):
        dados = {
            'pesos': self.pesos,
            'epoca': self.epoca,
            'taxa_aprendizado': self.taxa_aprendizado,
            'fator_normalizacao': self.fator_normalizacao,
        }

        util.salvar_dados(dados, caminho)

    def carregar(caminho):
        rn = Adaline()

        dados = util.carregar_dados(caminho)

        rn.pesos = dados['pesos']
        rn.epoca = dados['epoca']
        rn.taxa_aprendizado = dados['taxa_aprendizado']
        rn.fator_normalizacao = dados['fator_normalizacao']

        return rn