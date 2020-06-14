import pickle
import matplotlib.pyplot as plt

def normalizar(array_numpy):
    fator = max(abs(array_numpy[:,1:].min()), abs(array_numpy[:,1:].max()))

    fator = 12.071

    array_numpy[:,1:] /= fator

    return (array_numpy, fator)

def salvar_dados(dados, caminho):
    arq = open(caminho, 'wb')
    pickle.dump(dados, arq)
    arq.close()

def carregar_dados(caminho):
    arq = open(caminho, 'rb')
    dados = pickle.load(arq)
    arq.close()
    
    return dados

def plotar_dados_2d(pontos=[], retas=[], classes_pontos=None, mostrar=False, imagem_destino=None, limite='bipolar'):
    # limpar eventuais lixos no grafico
    plt.clf()

    # plotar pontos
    for i in range(len(pontos)):
        if(type(classes_pontos) == type(None) or classes_pontos[i] == 1):
            plt.plot(pontos[i,1],pontos[i,2],'b.')
        else:
            plt.plot(pontos[i,1],pontos[i,2],'g.')

    # plotar retas
    for r in retas:
        plt.plot(r[0], r[1])

    # configurar limites do grafico (os dados devem estar normalizados)
    if(limite == 'bipolar'):
        plt.xlim(-1,1)
        plt.ylim(-1,1)

    # salvar imagem
    if(imagem_destino != None):
        plt.savefig(imagem_destino)

    # mostrar
    if(mostrar):
        plt.show()
    
    plt.clf()

def plotar_dados_3d(
        pontos=[], 
        retas=[], 
        planos=[], 
        classes_pontos=None, 
        mostrar=False, 
        imagem_destino=None, 
        limite=None):
        
    # limpar eventuais lixos no grafico
    # plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plotar pontos
    for i in range(len(pontos)):
    # for i in range(1):
        if(type(classes_pontos) == type(None) or classes_pontos[i] == 1):
            ax.scatter(pontos[i,1], pontos[i,2], pontos[i,3], c='b')
        else:
            ax.scatter(pontos[i,1], pontos[i,2], pontos[i,3], c='g')

    # plotar retas
    for r in retas:
        plt.plot(r[0], r[1])
    
    # plotar planos
    for p in planos:
        ax.plot_surface(p[0], p[1], p[2])

    # configurar vizualizacao do grafico (especifico para o projeto em questao)
    # (dados normalizado)
    ax.set_xlim(-0.14,0.22)
    ax.set_ylim(-0.02,0.10)
    ax.set_zlim(0.1,1.1)
    ax.view_init(azim=61, elev=37)

    # salvar imagem
    if(imagem_destino != None):
        plt.savefig(imagem_destino)

    # mostrar
    if(mostrar):
        plt.show()
    
    plt.close('all')

def txt_para_dataset(arquivo_origem, arquivo_destino):
    x = []
    d = []

    while(True):
        try:
            dados = [float(i) for i in input().split(' ')]
        except EOFError:
            break

        x.append(dados[:-1])
        d.append(dados[-1])
    
    x = np.array(x)
    d = np.array(d)

    util.salvar_dados({'x':x,'d':d}, arquivo_destino)