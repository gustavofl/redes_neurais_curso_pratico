import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import shapely.geometry as sg
import shapely.ops as so

import util

class PlotRN_2D:

    def __init__(self, rn, buffer):
        self.rn = rn
        self.buffer = buffer
        self.patches_camada_anterior = []

    def plotar_curva_aprendizado(self, titulo=''):
        matplotlib.use('GTK3Agg')

        plt.plot(np.arange(len(self.rn.historico_erro)),self.rn.historico_erro)

        plt.title(titulo)
        plt.show()
        plt.close('all')
    
    def plotar_aprendizado(self, x=None, d=None, titulo=''):
        matplotlib.use('GTK3Agg')
        fig,ax = plt.subplots( max(self.rn.topologia[:1]) , len(self.rn.topologia)-1 )

        self.plotar_rede_neural(ax)

        plt.show()
        plt.close('all')
    
    def func_animacao(self, i, ax, para_salvar):
        if(para_salvar or 1):
            print('%d/%d'%(i,len(self.buffer)-1))
        
        poligonos_camada_anterior = []
        poligonos = ()
        for ci in range(len(self.rn.topologia)-1):
            poligonos_camada_atual = []
            for ni in range(max(self.rn.topologia[:1])):
                eixos = ax
                if(len(self.rn.topologia)-1 > 1):
                    eixos = ax[:,ci]
                
                if(para_salvar):
                    eixos[ni].clear()

                poligonos_neuronio = self.plotar_classes_aprendidas(eixos[ni], 1, ci, ni, poligonos_camada_anterior)
                poligonos += poligonos_neuronio
                poligonos_camada_atual.append(poligonos_neuronio)

                eixos[ni].set_xlim( -1.1 , 1.1 )
                eixos[ni].set_ylim( -1.1 , 1.1 )
            
            poligonos_camada_anterior = poligonos_camada_atual[:]
        
        return poligonos

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
    
    def plotar_rede_neural(self, ax):
        patches = ()
        if(len(self.rn.topologia)-1 == 1):
            patches += self.plotar_camada(ax, 0)
        else:
            for ind_camada in range(len(self.rn.topologia)-1):
                patches += self.plotar_camada(ax[:,ind_camada], ind_camada)
        
        return patches

    def plotar_camada(self, ax, ind_camada):
        patches = []
        for ind_neoronio in range(self.rn.topologia[ind_camada+1]):
            patches.append(self.plotar_neoronio(ax[ind_neoronio], ind_camada, ind_neoronio))

        self.patches_camada_anterior = patches[:]
        patches = tuple(np.array(patches).flatten())

        return patches
    
    def plotar_neoronio(self, ax, ind_camada, ind_neoronio):
        if(ind_camada == 0):
            poligonos = self.plotar_neoronio_linear(ax, ind_camada, ind_neoronio)
        else:
            poligonos = self.plotar_neoronio_nao_linear(ax, ind_camada, ind_neoronio)
        
        pesos = self.rn.camadas[ind_camada][ind_neoronio]
        patches = []
        for pol in poligonos:
            if(util.classificar_poligono(pol, pesos) > 0):
                patches += ax.fill(*pol.exterior.xy, c='blue', alpha=0.4)
            else:
                patches += ax.fill(*pol.exterior.xy, c='orange', alpha=0.4)

        ax.set_xlim( -1.1 , 1.1 )
        ax.set_ylim( -1.1 , 1.1 )

        return tuple(patches)
    
    def plotar_neoronio_linear(self, ax, ind_camada, ind_neoronio):
        t = 1.5
        pesos = self.rn.camadas[ind_camada][ind_neoronio]

        area = sg.Polygon([(-t,-t),(-t,t),(t,t),(t,-t),(-t,-t)])

        p1 = (-t,util.equacao_reduzida_reta(-t,pesos))
        p2 = (t,util.equacao_reduzida_reta(t,pesos))
        interseccao = sg.LineString([ p1 , p2 ])

        poligonos = so.split(area,interseccao)

        return poligonos
    
    def plotar_neoronio_nao_linear(self, ax, ind_camada, ind_neoronio):
        extremos = [np.array([-1,1]) for i in range(self.rn.topologia[ind_camada])]
        extremos = np.meshgrid(*extremos,)
        extremos = np.array([ext.flatten() for ext in extremos]).T
        extremos = np.concatenate((np.ones([extremos[:,0].size,1])*-1,extremos),axis=1)

        pesos = self.rn.camadas[ind_camada][ind_neoronio]
        classificacao_extremos = util.pseudo_neoronio(extremos, pesos)

        # uniao das influencias da camada anterior no neoronio atual
        uniao = None
        
        for i in range(classificacao_extremos.size):
            x = extremos[i][1:]
            y = classificacao_extremos[i]

            if(y == 1):
                # influencia do neoronio i da camada anterior no neoronio ind_neoronio da camada atual
                interseccao = None
    
                for j in range(len(self.patches_camada_anterior)):
                    pesos_camada_ant = self.rn.camadas[ind_camada-1][j]
                    neoronio = self.patches_camada_anterior[j]

                    # classe dos poligonos do neoronio atual que sera usado
                    classe_poligonos = x[j]

                    for k in range(len(neoronio)):
                        vertices = neoronio[k].get_path().vertices
                        pol = sg.Polygon(vertices)
                        
                        classe_poligono = util.classificar_poligono(pol, pesos_camada_ant)

                        if(classe_poligono == classe_poligonos):
                            if(interseccao == None):
                                interseccao = pol
                            else:
                                interseccao = interseccao.intersection(pol)
            
                if(uniao == None):
                    uniao = interseccao
                else:
                    uniao = uniao.union(interseccao)
        
        poligono_nao_classificado = sg.Polygon([(-1.5,-1.5),(-1.5,1.5),(1.5,1.5),(1.5,-1.5)])
        poligono_nao_classificado = poligono_nao_classificado.symmetric_difference(uniao)

        return (uniao, poligono_nao_classificado)
    
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
