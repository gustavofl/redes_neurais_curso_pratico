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

        self.plotar_rede_neural(ax, self.rn.camadas)

        plt.show()
        plt.close('all')
    
    def func_animacao(self, i, ax, para_salvar):
        if(para_salvar or 1):
            print('%d/%d'%(i,len(self.buffer)-1))
        
        pesos = self.buffer[i]
        patches = self.plotar_rede_neural(ax, pesos)

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
    
    def plotar_rede_neural(self, ax, pesos):
        patches = ()
        if(len(self.rn.topologia)-1 == 1):
            patches += self.plotar_camada(ax, 0, pesos)
        else:
            for ind_camada in range(len(self.rn.topologia)-1):
                patches += self.plotar_camada(ax[:,ind_camada], ind_camada, pesos)
            
        return patches

    def plotar_camada(self, ax, ind_camada, pesos):
        patches_camada = ()
        patches_neoronios = []

        for ind_neoronio in range(self.rn.topologia[ind_camada+1]):
            patches_neoronio = self.plotar_neoronio(ax[ind_neoronio], ind_camada, ind_neoronio, pesos)
            patches_camada += patches_neoronio
            patches_neoronios.append(patches_neoronio)

        self.patches_camada_anterior = patches_neoronios[:]

        return patches_camada
    
    def plotar_neoronio(self, ax, ind_camada, ind_neoronio, pesos):
        if(ind_camada == 0):
            poligonos = self.plotar_neoronio_linear(ax, ind_camada, ind_neoronio, pesos)
        else:
            poligonos = self.plotar_neoronio_nao_linear(ax, ind_camada, ind_neoronio, pesos)

        pesos = pesos[ind_camada][ind_neoronio]
        patches = []
        for pol in poligonos:
            if(not pol.is_empty):
                if(util.classificar_poligono(pol, pesos) > 0):
                    patches += ax.fill(*pol.exterior.xy, c='blue', alpha=0.4)
                else:
                    patches += ax.fill(*pol.exterior.xy, c='orange', alpha=0.4)

        ax.set_xlim( -1.1 , 1.1 )
        ax.set_ylim( -1.1 , 1.1 )

        return tuple(patches)
    
    def plotar_neoronio_linear(self, ax, ind_camada, ind_neoronio, pesos):
        t = 1.5
        pesos = pesos[ind_camada][ind_neoronio]

        area = sg.Polygon([(-t,-t),(-t,t),(t,t),(t,-t),(-t,-t)])

        p1 = (-t,util.equacao_reduzida_reta(-t,pesos))
        p2 = (t,util.equacao_reduzida_reta(t,pesos))
        interseccao = sg.LineString([ p1 , p2 ])

        poligonos = so.split(area,interseccao)

        return poligonos
    
    def plotar_neoronio_nao_linear(self, ax, ind_camada, ind_neoronio, pesos):
        extremos = [np.array([-1,1]) for i in range(self.rn.topologia[ind_camada])]
        extremos = np.meshgrid(*extremos,)
        extremos = np.array([ext.flatten() for ext in extremos]).T
        extremos = np.concatenate((np.ones([extremos[:,0].size,1])*-1,extremos),axis=1)

        pesos_neoronio_atual = pesos[ind_camada][ind_neoronio]
        classificacao_extremos = util.pseudo_neoronio(extremos, pesos_neoronio_atual)

        quadrado_cheio = sg.Polygon([(-1.5,-1.5),(-1.5,1.5),(1.5,1.5),(1.5,-1.5)])

        # uniao das influencias da camada anterior no neoronio atual
        uniao = sg.Polygon([])
        
        for i in range(classificacao_extremos.size):
            x = extremos[i][1:]
            y = classificacao_extremos[i]

            if(y == 1):
                # influencia do neoronio i da camada anterior no neoronio ind_neoronio da camada atual
                interseccao = quadrado_cheio
    
                for j in range(len(self.patches_camada_anterior)):
                    pesos_camada_ant = pesos[ind_camada-1][j]
                    neoronio = self.patches_camada_anterior[j]

                    # classe dos poligonos do neoronio atual que sera usado
                    classe_poligonos = x[j]

                    for k in range(len(neoronio)):
                        vertices = neoronio[k].get_path().vertices
                        pol = sg.Polygon(vertices)
                        
                        classe_poligono = util.classificar_poligono(pol, pesos_camada_ant)

                        if(classe_poligono == classe_poligonos):
                            interseccao = interseccao.intersection(pol)
            
                uniao = uniao.union(interseccao)
        
        poligono_nao_classificado = quadrado_cheio
        poligono_nao_classificado = poligono_nao_classificado.symmetric_difference(uniao)

        poligonos = []
        if(not uniao.is_empty):
            poligonos.append(uniao)
        if(not poligono_nao_classificado.is_empty):
            poligonos.append(poligono_nao_classificado)
        
        return poligonos
    
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
