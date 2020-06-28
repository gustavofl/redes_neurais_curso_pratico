import math
import numpy as np

from plot_rn_2d import PlotRN_2D
import util

class PMC:

    def __init__(self, taxa_aprendizado=0.001, topologia=[2,2]):
        self.epoca = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_normalizacao = None

        self.topologia = topologia
        self.vetores_forward = [] # usado no backpropagation
        self.camadas = []
        for i in range(1,len(topologia)):
            self.vetores_forward.append((None,None))
            self.camadas.append(np.random.rand(topologia[i],topologia[i-1]+1))
        
        self.historico_camadas = []
        self.historico_erro = []
        self.plot_rn = PlotRN_2D(self, self.historico_camadas)
    
    def forward(self, x):
        for i in range(len(self.camadas)):
            pesos = self.camadas[i]

            x = np.concatenate((np.ones([x[:,0].size,1])*-1,x),axis=1)
            I = x.dot(pesos.T)
            Y = self.ativacao(I)
            x = Y

            self.vetores_forward[i] = (I,Y)
        
        return Y
    
    def backward(self, x, d):
        grad_camada_posterior = 0
        
        for i in range(len(self.camadas)-1,-1,-1):
            # Verificando saida da camada anterior
            if(i == 0):
                Y_ant = x
            else:
                Y_ant = self.vetores_forward[i-1][1]
            Y_ant = np.concatenate((np.ones([Y_ant[:,0].size,1])*-1,Y_ant),axis=1)
            
            # calculando gradientes da camada atual
            I,Y = self.vetores_forward[i]
            if(i == len(self.camadas)-1):
                grad_local = (d-Y) * self.derivada_ativacao(I)
            else:
                grad_local = grad_camada_posterior.dot(self.camadas[i+1][:,1:]) * self.derivada_ativacao(I)
            
            # Armazenando gradiente para calculo na camada anterior
            grad_camada_posterior = grad_local

            # Atualizando pesos da camada atual
            self.camadas[i] += self.taxa_aprendizado * grad_local.T.dot(Y_ant)

    def ativacao(self, u, beta=0.5):
        # return 1/(1+math.e**(-beta*u)) # logistica
        return (math.e**(beta*u) - math.e**(-beta*u)) / (math.e**(beta*u) + math.e**(-beta*u)) # tanh
    
    def derivada_ativacao(self, u, beta=0.5):
        # return beta * self.ativacao(u,beta) * (1 - self.ativacao(u,beta)) # logistica
        return beta * (1 - self.ativacao(u,beta)**2) # tanh

    def treinar(self, x, d, verbose=False, guardar_historico=False):
        x = np.copy(x)

        x,self.fator_normalizacao = util.normalizar(x, self.fator_normalizacao)

        erro_minimo = 1e-12
        ultimo_erro = 100
        while(self.epoca <= 2000):
            y = self.forward(x)

            erro = ((d-y)**2).sum()/float(d.size)
            
            if(verbose):
                print('%04d - %.12f'%(self.epoca, erro))

            if(ultimo_erro-erro <= erro_minimo):
                print('UNDERFITTING')
                break

            self.backward(x,d)

            ultimo_erro = erro
            self.epoca += 1

            if(guardar_historico and self.epoca%1 == 0):
                pesos = [np.copy(camada) for camada in self.camadas]
                self.historico_camadas.append(pesos)
                self.historico_erro.append(erro)

    def classificar(self, x, salvar_imagem=False, x_normalizado=False):
        x = np.copy(x)

        if(not x_normalizado):
            x,fator = util.normalizar(x,self.fator_normalizacao)
        
        u = self.forward(x)
        y = self.ativacao(u)

        return y
    
    def plotar_animacao(self, x=None, d=None, titulo=''):
        if(len(self.historico_camadas) > 0):
            self.plot_rn.plotar_animacao(x, d, titulo=titulo)
    
    def salvar_animacao(self, x=None, d=None, titulo='', nome_arquivo=''):
        if(len(self.historico_camadas) > 0):
            self.plot_rn.salvar_animacao(x, d, titulo=titulo, nome_arquivo=nome_arquivo)

    def plotar_aprendizado(self, x=None, d=None, titulo=''):
        self.plot_rn.plotar_aprendizado(x, d, titulo=titulo)

    def plotar_curva_aprendizado(self, titulo=''):
        self.plot_rn.plotar_curva_aprendizado(titulo=titulo)
    
    def salvar(self, caminho):
        dados = {
            'camadas': self.camadas,
            'epoca': self.epoca,
            'taxa_aprendizado': self.taxa_aprendizado,
            'fator_normalizacao': self.fator_normalizacao,
        }

        util.salvar_dados(dados, caminho)

    def carregar(caminho):
        rn = PMC()

        dados = util.carregar_dados(caminho)

        rn.camadas = dados['camadas']
        rn.epoca = dados['epoca']
        rn.taxa_aprendizado = dados['taxa_aprendizado']
        rn.fator_normalizacao = dados['fator_normalizacao']

        return rn