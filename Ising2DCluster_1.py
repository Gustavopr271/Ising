import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
start=time.time()

L = 8
N = L**2
nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
            (i // L) * L + (i - 1) % L, (i - L) % N)
                                    for i in range(N)}
e=1
h=0
T = list_T = [1.4 + 0.05 * i for i in range(40)]

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

class Test:
    def __init__(self, limit):
        self.limit = limit

    def __iter__(self):
        self.x = [np.random.choice([1, -1]) for k in range(N)]
        self.y=1
        return self

    def __next__(self):
        x = self.x
        y = self.y

        if y > self.limit:
            raise StopIteration

        self.x = clus(self.x)
        self.y = y+1
        return x
t_cv=[]
t_susc=[]
t_meanmag=[]
t_meanenergy=[]
for temp in T:
    p=1.0 - np.exp(-2*e/ temp)
    l_energy=[]
    l_mag=[]
    it=500
    for i in Test(it):
        l_energy.append(energy(i,N,nbr))
        l_mag.append(mag(i))
    mean_energy= np.mean(l_energy)
    mean_mag=np.mean(l_mag)
    mean_sqr_energy= np.mean([x**2 for x in l_energy])
    mean_sqr_mag= np.mean([x**2 for x in l_mag])
    t_meanmag.append(mean_mag)
    t_meanenergy.append(mean_energy)
    t_cv.append((1/temp**2)*(mean_sqr_energy-mean_energy**2)/N)
    t_susc.append((1/temp)*(mean_sqr_mag-mean_mag**2)/N)
end = time.time()
print('tiempo',(end-start))

dict={'Temp':T, 'c_v':t_cv, 'MagSusc':t_susc, 'MeanMag':t_meanmag, 'MeanEnergy':t_meanenergy}
df=pd.DataFrame(dict)
df.to_csv(r'C:\Users\GUSTAVO\Desktop\PY4E\GraphThesisLic\Final_Scripts\2d_cvsus_20.csv', index=False)

plt.subplot(2,1,1)
plt.scatter(T, t_cv, s=5)
plt.ylabel("Calor especifico")
plt.legend([str(L)+'x'+str(L)])

plt.subplot(2,1,2)
plt.scatter(T, t_susc, s=5)
plt.xlabel("Temperatura")
plt.ylabel("Susceptibilidad magnetica")
plt.savefig(r"C:\Users\GUSTAVO\Desktop\PY4E\GraphThesisLic\Final_Scripts\2d_cvsus_20.jpg")
plt.show()
