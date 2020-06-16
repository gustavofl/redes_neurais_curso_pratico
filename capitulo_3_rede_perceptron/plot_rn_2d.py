import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

import util

class PlotRN_2D:

    def __init__(self, rn):
        self.rn = rn

    def plotar_classes_aprendidas(self, ax, tamanho):
        pol_1,pol_2 = self.poligonos_classes(tamanho)

        patch_1 = patches.Polygon(pol_1, color='orange', alpha=0.4, zorder=0)
        patch_2 = patches.Polygon(pol_2, color='blue', alpha=0.4, zorder=0)

        ax.add_patch(patch_1)
        ax.add_patch(patch_2)
    
    def plotar_dataset(self, x, y):
        if(type(y) != type(None)):
            # laranja   = (255,160,0)   = ( 1 , 0.6 , 0  )
            # azul      = (0,0,255)     = ( 0 , 0   , 1. )
            cores = np.empty([y.size,3])
            cores[:,0] = (1-y)/2
            cores[:,1] = (1-y)/3
            cores[:,2] = (y+1)/2
        else:
            cores = np.zeros([y.size,3])
        
        plt.scatter(x[:,1], x[:,2], color=cores, s=10, zorder=1)
    
    def plotar_aprendizado(self, x=None, d=None, mostrar=False, imagem_destino=None, titulo=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        maior_valor_absoluto_dataset = 1
        
        if(type(x) != type(None)):
            x = np.copy(x)

            x,fator = util.normalizar(x, self.rn.fator_normalizacao)

            self.plotar_dataset(x, d)

            maior_valor_absoluto_dataset = util.maior_valor_absoluto(x)

        self.plotar_classes_aprendidas(ax, maior_valor_absoluto_dataset)

        ax.set_xlim( -maior_valor_absoluto_dataset*1.1 , maior_valor_absoluto_dataset*1.1)
        ax.set_ylim( -maior_valor_absoluto_dataset*1.1 , maior_valor_absoluto_dataset*1.1)

        plt.title(titulo)
        plt.show()

        plt.close('all')
    
    def poligonos_classes(self, tamanho):
        tamanho *= self.rn.fator_normalizacao*1.5

        x = np.empty([4,3])
        x[:,0] = -1
        x[:,1] = np.array([ -tamanho , -tamanho , tamanho ,  tamanho ])
        x[:,2] = np.array([ -tamanho ,  tamanho , tamanho , -tamanho ])

        y = self.rn.classificar(x)

        poligonos = [[],[]]

        indice_area_atual = (y[-1]+1)//2
        classe_atual = y[-1]
        poligonos[indice_area_atual].append(x[-1,1:])

        for xi,yi in zip(x,y):

            if(yi != classe_atual):
                if(xi[1] != xi[2]):
                    intersecsao_reta = [ xi[1] , self.f(xi[1]) ]
                else:
                    intersecsao_reta = [ self.f_inversa(xi[2]) , xi[2] ]

                poligonos[0].append(intersecsao_reta)
                poligonos[1].append(intersecsao_reta)

                indice_area_atual = (yi+1)//2

                classe_atual = yi
            
            poligonos[indice_area_atual].append(xi[1:])
        
        return poligonos
    
    def f(self, x):
        # equacao reduzida da reta encontrada pela rn
        w0,w1,w2 = self.rn.pesos

        return -w1/w2 * x + w0/w2
    
    def f_inversa(self, y):
        # equacao reduzida inversa da reta encontrada pela rn
        w0,w1,w2 = self.rn.pesos

        return -w2/w1 * y + w0/w1