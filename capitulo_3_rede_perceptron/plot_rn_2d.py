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
    
    def plotar_aprendizado(self, x=None, d=None, titulo=''):
        matplotlib.use('GTK3Agg')
        fig,ax,maior_valor_absoluto_dataset = self.iniciar_figura_e_plotar_dataset(x, d, titulo)

        self.plotar_classes_aprendidas(ax, maior_valor_absoluto_dataset, self.rn.pesos)

        plt.show()
        plt.close('all')

    def plotar_animacao(self, x=None, d=None, titulo=''):
        matplotlib.use('GTK3Agg')
        fig,ax,maior_valor_absoluto_dataset = self.iniciar_figura_e_plotar_dataset(x, d, titulo)
    
        def func_animacao(i=0):
            pesos = self.buffer[i]
            return self.plotar_classes_aprendidas(ax, 1, pesos)

        animacao = animation.FuncAnimation(fig, func_animacao, frames=len(self.buffer), 
                                                init_func=func_animacao, interval=50, blit=True, repeat=False)
        
        plt.show()
        plt.close('all')

    def salvar_animacao(self, x=None, d=None, titulo='', nome_arquivo=''):
        matplotlib.use('Agg')
        fig,ax,maior_valor_absoluto_dataset = self.iniciar_figura_e_plotar_dataset(x, d, titulo)
    
        def func_animacao(i=0):
            pesos = self.buffer[i]

            func_animacao.patches[0].remove()
            func_animacao.patches[1].remove()
            func_animacao.patches = self.plotar_classes_aprendidas(ax, 1, pesos)

            if((i+1)%(len(self.buffer)//4) == 0 or i == len(self.buffer)-1):
                print('%d%%'%int(100*(i+1)/float(len(self.buffer))))

            return func_animacao.patches

        pesos = self.buffer[0]
        func_animacao.patches = self.plotar_classes_aprendidas(ax, 1, pesos)

        print('Salvando animação em arquivo')

        animacao = animation.FuncAnimation(fig, func_animacao, frames=len(self.buffer), 
                                            interval=50, blit=True, repeat=False)
        
        animacao.save(nome_arquivo)
    
    def iniciar_figura_e_plotar_dataset(self, x, y, titulo):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        maior_valor_absoluto_dataset = 1
        
        if(type(x) != type(None)):
            x = np.copy(x)

            x,fator = util.normalizar(x, self.rn.fator_normalizacao)

            #### CALCULAR CORES
            if(type(y) != type(None)):
                # laranja   = (255,160,0)   = ( 1 , 0.6 , 0  )
                # azul      = (0,0,255)     = ( 0 , 0   , 1. )
                cores = np.empty([y.size,3])
                cores[:,0] = (1-y)/2
                cores[:,1] = (1-y)/3
                cores[:,2] = (y+1)/2
            else:
                cores = np.zeros([y.size,3])
            
            #### PLOTAR PONTOS
            plt.scatter(x[:,1], x[:,2], color=cores, s=10, zorder=1)

            maior_valor_absoluto_dataset = util.maior_valor_absoluto(x)

        ax.set_xlim( -maior_valor_absoluto_dataset*1.1 , maior_valor_absoluto_dataset*1.1)
        ax.set_ylim( -maior_valor_absoluto_dataset*1.1 , maior_valor_absoluto_dataset*1.1)

        plt.title(titulo)

        return fig,ax,maior_valor_absoluto_dataset
    
    def plotar_classes_aprendidas(self, ax, tamanho, pesos):
        pol_1,pol_2 = self.poligonos_classes(tamanho, pesos)

        patch_1 = patches.Polygon(pol_1, color='orange', alpha=0.4, zorder=0)
        patch_2 = patches.Polygon(pol_2, color='blue', alpha=0.4, zorder=0)

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
