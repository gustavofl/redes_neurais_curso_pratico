import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import util

class PlotRN_2D:

    def __init__(self, rn, buffer):
        self.rn = rn
        self.buffer = buffer
    
    def plotar_dataset(self, x, y=None):
        matplotlib.use('GTK3Agg')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = np.copy(x)

        x,fator = util.normalizar(x, self.rn.fator_normalizacao)

        #### CALCULAR CORES
        if(type(y) != type(None)):
            y = y[:,1]

            # laranja   = (255,160,0)   = ( 1 , 0.6 , 0  )
            # azul      = (0,0,255)     = ( 0 , 0   , 1. )
            cores = np.empty([y.size,3])
            cores[:,0] = (1-y)/2
            cores[:,1] = (1-y)/3
            cores[:,2] = (y+1)/2
        else:
            cores = np.zeros([y.size,3])

        
        #### PLOTAR PONTOS
        plt.scatter(x[:,0], x[:,1], color=cores, s=10)

        ax.set_xlim( -1.1 , 1.1)
        ax.set_ylim( -1.1 , 1.1)

        plt.show()
        plt.close('all')

    def plotar_curva_aprendizado(self, titulo=''):
        matplotlib.use('GTK3Agg')

        plt.plot(np.arange(len(self.rn.historico_erro)),self.rn.historico_erro)

        plt.title(titulo)
        plt.show()
        plt.close('all')
    
    def plotar_aprendizado(self, x=None, d=None, titulo=''):
        matplotlib.use('GTK3Agg')
        fig,ax = plt.subplots( max(self.rn.topologia[:1]) , len(self.rn.topologia)-1 )

        for ci in range(len(self.rn.topologia)-1):
            for ni in range(max(self.rn.topologia[:1])):
                eixos = ax
                if(len(self.rn.topologia)-1 > 1):
                    eixos = ax[ci]

                self.plotar_classes_aprendidas(eixos[ni], 1, self.rn.camadas[ci][ni])
                ax[ni].set_xlim( -1.1 , 1.1 )
                ax[ni].set_ylim( -1.1 , 1.1 )

        plt.show()
        plt.close('all')
    
    def func_animacao(self, i, ax, para_salvar):
        camadas = self.buffer[i]

        if(para_salvar or 1):
            print('%d/%d'%(i,len(self.buffer)-1))
        
        patches = ()
        for ci in range(len(self.rn.topologia)-1):
            for ni in range(max(self.rn.topologia[:1])):
                eixos = ax
                if(len(self.rn.topologia)-1 > 1):
                    eixos = ax[:,ci]
                
                if(para_salvar):
                    eixos[ni].clear()

                patches += self.plotar_classes_aprendidas(eixos[ni], 1, camadas[ci][ni])
                eixos[ni].set_xlim( -1*1.1 , 1*1.1 )
                eixos[ni].set_ylim( -1*1.1 , 1*1.1 )
            
        return patches

    def plotar_animacao(self, x=None, d=None, titulo=''):
        matplotlib.use('GTK3Agg')
        fig,ax = plt.subplots( max(self.rn.topologia[:1]) , len(self.rn.topologia)-1 )

        animacao = animation.FuncAnimation(fig, self.func_animacao, frames=len(self.buffer), 
                                            fargs=(ax,False,), interval=50, blit=True, repeat=False)
        
        plt.show()
        plt.close('all')

    def salvar_animacao(self, x=None, d=None, titulo='', nome_arquivo=''):
        matplotlib.use('Agg')
        fig,ax = plt.subplots( max(self.rn.topologia[:1]) , len(self.rn.topologia)-1 )
    
        print('Salvando animação em arquivo')

        animacao = animation.FuncAnimation(fig, self.func_animacao, frames=len(self.buffer), 
                                            fargs=(ax,True,), interval=50, blit=True, repeat=False)
        
        animacao.save(nome_arquivo)
    
    def plotar_classes_aprendidas(self, ax, tamanho, pesos):
        pol_1,pol_2 = self.poligonos_classes(tamanho, pesos)

        patch_1 = patches.Polygon(pol_1, color='orange', alpha=0.4)
        patch_2 = patches.Polygon(pol_2, color='blue', alpha=0.4)

        ax.add_patch(patch_1)
        ax.add_patch(patch_2)

        return patch_1,patch_2,
    
    def poligonos_classes(self, tamanho, pesos):
        tamanho *= self.rn.fator_normalizacao*1.5

        x = np.empty([4,3])
        x[:,0] = -1
        x[:,1] = np.array([ -tamanho , -tamanho , tamanho ,  tamanho ])
        x[:,2] = np.array([ -tamanho ,  tamanho , tamanho , -tamanho ])

        y = self.pseudo_classificar(x, pesos)

        poligonos = [[],[]]

        indice_area_atual = (y[-1]+1)//2
        classe_atual = y[-1]
        poligonos[indice_area_atual].append(x[-1,1:])

        for xi,yi in zip(x,y):

            if(yi != classe_atual):
                if(xi[1] != xi[2]):
                    intersecsao_reta = [ xi[1] , self.f(xi[1], pesos) ]
                else:
                    intersecsao_reta = [ self.f_inversa(xi[2], pesos) , xi[2] ]

                poligonos[0].append(intersecsao_reta)
                poligonos[1].append(intersecsao_reta)

                indice_area_atual = (yi+1)//2

                classe_atual = yi
            
            poligonos[indice_area_atual].append(xi[1:])
        
        return poligonos

    def pseudo_classificar(self, x, pesos):
        # simula a classificacao da rn a partir dos pesos armazenados
        
        # calculo com os pesos
        u = x.dot(np.transpose(pesos))

        # ativacao (degrau bipolar)
        y = (u>=0)*2-1

        return y
    
    def f(self, x, pesos):
        # equacao reduzida da reta encontrada pela rn
        w0,w1,w2 = pesos

        return -w1/w2 * x + w0/w2
    
    def f_inversa(self, y, pesos):
        # equacao reduzida inversa da reta encontrada pela rn
        w0,w1,w2 = pesos

        return -w2/w1 * y + w0/w1
