import random, math, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#This program is used to compute mean energy and magnetic susceptibility
import time
start=time.time()

L = 50
N = L * L
nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
            (i // L) * L + (i - 1) % L, (i - L) % N)
                                    for i in range(N)}
e=1
h=0
T_1 = [1.65+0.01*i for i in range(60)]   ###behaviour near below to Tc
T_2 = [2.29+0.01*i for i in range(60)]   ###behaviour near up to Tc
T=T_1+T_2

nsteps = 12000
S = [1 for k in range(N)]

def energy(S, N, nbr):
    E = 0.0
    for k in range(N):
        E -=  e*S[k] * sum(S[nn] for nn in nbr[k]) + 2*h*S[k]
    return (0.5 * E)

def mag(S):
    mag=abs(np.sum(S))
    return(mag)

def clus(S):
    k=np.random.randint(0,N-1)
    Pocket, Cluster = [k], [k]
    while Pocket != []:
        j = np.random.choice(Pocket)
        for l in nbr[j]:
            if S[l] == S[j] and l not in Cluster and np.random.uniform(0.0, 1.0) < p:
                Pocket.append(l)
                Cluster.append(l)
        Pocket.remove(j)
    for j in Cluster:
        S[j] *= -1
    return S

list_cv=[]
list_meanmag=[]
list_meanenergy=[]
list_suscep=[]
for temp in T:
    p  = 1.0 - np.exp(-2.0 / temp)
    l_energy=[]
    l_mag=[]
    l_sqrenergy=[]
    l_sqrmag=[]
    for step in range(nsteps):
        S=clus(S)
        l_energy.append(energy(S,N,nbr))
        l_mag.append(mag(S))
    mean_energy= np.mean(l_energy)
    mean_mag= np.mean(l_mag)
    suscep=(1/temp)*(np.mean([j**2 for j in l_mag])-mean_mag**2)/N
    c_v=(1/temp**2)*(np.mean([j**2 for j in l_energy])-mean_energy**2)/N
    list_cv.append(c_v)
    list_meanmag.append(mean_mag)
    list_meanenergy.append(mean_energy)
    list_suscep.append(suscep)
end = time.time()
print('tiempo',(end-start))

dict={'Temp':T, 'c_v':list_cv, 'MagSusc':list_suscep}
df=pd.DataFrame(dict)
df.to_csv(r'C:\Users\GUSTAVO\Desktop\PY4E\GraphThesisLic\Final_Scripts\2d_cvsus_20.csv', index=False)

plt.subplot(2,1,1)
plt.scatter(T, list_cv, s=5)
plt.ylabel("Calor especifico")
plt.legend([str(L)+'x'+str(L)])

plt.subplot(2,1,2)
plt.scatter(T, list_suscep, s=5)
plt.xlabel("Temperatura")
plt.ylabel("Susceptibilidad magnetica")
plt.savefig(r"C:\Users\GUSTAVO\Desktop\PY4E\GraphThesisLic\Final_Scripts\2d_cvsus_20.jpg")
plt.show()
