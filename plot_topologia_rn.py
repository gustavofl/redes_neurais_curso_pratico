import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import util

def equacao_reta(x, y):
    # RETA: y = 0.25*x + 0.5
    return y > -1*x + 5

def ativacao_reta(x, y):
    # degrau bipolar
    return equacao_reta(x,y)*2-1

# util.gerar_dataset('reta_1/dataset_map.data', ativacao_reta, tamanho=1000)

x,d = util.carregar_dataset('reta_1/dataset_map.data')
x,f = util.normalizar(x, None)
d = np.array([d])

m1 = np.ones([x[:,0].size,1])

t = 1.1
s = 600.0

xx,yy = np.meshgrid( np.arange(0.6,t,t/s) , np.arange(0.6,t,t/s) )

md = np.empty([x[:,0].size,xx.size])
md[:] = d.T

pesos = np.empty([xx.size,3])
pesos[:,1] = xx.flatten()
pesos[:,2] = yy.flatten()

for w0 in np.arange(0.3,0.4,t/s):
    plt.clf()

    pesos[:,0] = w0
    
    u = x.dot(pesos.T)
    z = ((u-md)**2).T.dot(m1)/float(x[:,0].size)
    z = np.reshape(z, xx.shape)

    plt.contourf(xx,yy,z,50)
    plt.colorbar()

    plt.title('W0 = %.3f'%w0)
    plt.pause(0.01)