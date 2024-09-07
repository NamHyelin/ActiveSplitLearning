import numpy as np
import math
import matplotlib.pyplot as plt


def x(t, M, A, w, snr):
    result=0
    ph = np.random.uniform(0, math.pi * 2, M)
    for i in range(M):
        result += A[i]* math.cos(w[i]*t + ph[i])
    result+= np.random.normal(0, 1, M)

    return result/snr



def mle(x, M):
    '''frequency that maximizes the amplitude of the Fourier transform of the data'''
    output=0.0
    w=np.random.normal(0, 2*math.pi, 80)
    w_result=np.ones_like(w)
    for j in range(len(w)):
        for i in range(M):
            output += x[i]*np.exp(-1j*w[j]*i)
        w_result[j]=output
    max_w_idx= np.argmax(w_result)
    return w[max_w_idx]


def crb(A, sigma_2, N):
    '''frequency CRB'''
    result=0.0
    for i in range(len(A)):
        result+= (24  * (sigma_2[i])) / ((A[i]**2 * N**3))
    return result

'''Number 1'''
#case 1
t=200
M=1
A=[1]
w=[math.pi/20]

MSE=[]
CRB=[]
for sigma_2 in [1,5,10,15,20]:
    x_= x(t, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= math.pi/20
    crb_= crb(A, [sigma_2], t)
    MSE.append(10*math.log10(abs(w_max-w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()


#case 2
t=200
M=1
A=[1]
w=[2*math.pi/81]

MSE=[]
CRB=[]
for sigma_2 in [1,5,10,15,20]:
    x_= x(t, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= 2*math.pi/81
    crb_= crb(A, [sigma_2], t)
    MSE.append(10*math.log10(abs(w_max, w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()


#case 3
t=200
M=2
A=[1,1]
w=[2*math.pi/81, math.pi/2]

MSE=[]
CRB=[]
for sigma_2 in [0,5,10,15,20]:
    x_= x(t, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= 2*math.pi/81
    crb_= crb(A, [sigma_2], t)
    MSE.append(10*math.log10(abs(w_max, w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()





'''Number 2'''

#case 1
M=1
A=[1]
w=[7*math.pi/31]
sigma_2= 1

MSE=[]
CRB=[]
for N in [0,50,100,150,200,250,300]:
    x_= x(N, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= 7*math.pi/31
    crb_= crb(A, sigma_2, N)
    MSE.append(10*math.log10(abs(w_max, w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()


#case 2
M=1
A=[1]
w=[math.pi/2]
sigma_2= 1

MSE=[]
CRB=[]
for N in [0,50,100,150,200,250,300]:
    x_= x(N, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= 7*math.pi/31
    crb_= crb(A, sigma_2, N)
    MSE.append(10*math.log10(abs(w_max, w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()

#case 3
M=2
A=[1,1]
w=[7*math.pi/31, math.pi/2]
sigma_2= 1

MSE=[]
CRB=[]
for N in [0,50,100,150,200,250,300]:
    x_= x(N, M, A, w, sigma_2)
    w_max= mle(x_,M)
    w_truth= 7*math.pi/31
    crb_= crb(A, sigma_2, N)
    MSE.append(10*math.log10(abs(w_max, w_truth)))
    CRB.append(10*math.log10(crb_))
plt.plot(MSE, 'o-', label='MSE')
plt.plot(CRB, '--', label='CRB')
plt.grid()
plt.legend()
plt.show()