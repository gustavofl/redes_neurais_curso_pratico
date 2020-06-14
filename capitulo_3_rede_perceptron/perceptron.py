import numpy as np
import matplotlib.pyplot as plt

import util

class Perceptron:

    def __init__(self, qnt_entradas=3, taxa_aprendizado=0.01):
        self.pesos = np.random.rand(qnt_entradas)
        self.epoca = 0
        self.taxa_aprendizado = taxa_aprendizado
    
    def feedforward(self, x):
        return x.dot(np.transpose(self.pesos))

    def ativacao(self, u):
        # degrau bipolar
        return (u>=0)*2-1
    
    def construir_reta_aprendida(self):
        w0,w1,w2 = self.pesos

        r_x = [-2,2]
        r_y = [0,0]

        r_y[0] = -w1/w2 * r_x[0] + w0/w2
        r_y[1] = -w1/w2 * r_x[1] + w0/w2

        return r_x,r_y
    
    def construir_plano_aprendido(self):
        xx,yy = np.meshgrid( np.arange(-0.14,0.22,0.01) , np.arange(-0.02,0.10,0.01) )

        w0,w1,w2,w3 = self.pesos

        z = ( - w1/w3 * xx - w2/w3 * yy + w0/w3 )

        return xx,yy,z

    def salvar(self, caminho):
        dados = {
            'pesos': self.pesos,
            'epoca': self.epoca,
            'taxa_aprendizado': self.taxa_aprendizado,
        }

        util.salvar_dados(dados, caminho)

    def carregar(caminho):
        rn = Perceptron()

        dados = util.carregar_dados(caminho)

        rn.pesos = dados['pesos']
        rn.epoca = dados['epoca']
        rn.taxa_aprendizado = dados['taxa_aprendizado']

        return rn

    def treino(self, x, d, verbose=False, salvar_imagens=False):
        x,fator = util.normalizar(x)

        possui_erro = True
        while(possui_erro and self.epoca <= 2000):
            possui_erro = False
            
            u = self.feedforward(x)
            y = self.ativacao(u)

            if((y != d).any()):
                self.pesos = self.pesos + self.taxa_aprendizado * (d - y).dot(x)
                possui_erro = True

            if(verbose):
                erro = (d - y).dot(x)
                print(self.epoca, erro.sum(), erro)
            
            """ if(salvar_imagens and self.epoca%10 == 0):
                reta_aprendida = self.construir_reta_aprendida()

                util.plotar_dados_2d(
                    pontos=x, 
                    retas=[reta_aprendida], 
                    classes_pontos=d, 
                    imagem_destino='projeto/treinamento/epoca_%04d.png'%self.epoca) """
            
            if(salvar_imagens and self.epoca%10 == 0):
                plano = self.construir_plano_aprendido()

                util.plotar_dados_3d(
                    pontos=x, 
                    classes_pontos=d, 
                    planos=[plano], 
                    imagem_destino='projeto/treinamento/epoca_%04d.png'%self.epoca,
                    limite=None)
            
            self.epoca += 1
    
    def classificar(self, x, salvar_imagem=False):
        x,fator = util.normalizar(x)

        u = self.feedforward(x)
        y = self.ativacao(u)

        return y