import numpy as np
import math
from dataset import ti,yi
#------------------------------------------------------------------------------
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
#------------------------------------------------------------------------------
def hyperbolicIO(t,x):
    S = int((len(x)-1)/3)
    yhat = []
    for ti in t:
        toplam = x[3*S]
        for j in range(0,S):
            toplam += x[2*S+j]*tanh(x[j]*ti+x[S+j])
        yhat.append(toplam)
    return yhat
#------------------------------------------------------------------------------
def error(xk,ti,yi):
    yhat = hyperbolicIO(ti,xk)
    return np.array(yi) - np.array(yhat)
#------------------------------------------------------------------------------
def findJacobian(traininginput,x):
    S = int((len(x)-1)/3)
    numofdata = len(traininginput)    
    J = np.matrix(np.zeros((numofdata,3*S+1)))
    for i in range(0,numofdata):
        for j in range(0,S):
            J[i,j] = -x[j+2*S]*traininginput[i]*(1-tanh(x[j]*traininginput[i]+x[S+j])**2)
        for j in range(S,2*S):
            J[i,j] = -x[j+S]*(1-tanh(x[j-S]*traininginput[i]+x[j])**2)
        for j in range(2*S,3*S):
            J[i,j] = -tanh(x[j-2*S]*traininginput[i]+x[j-S])  
        J[i,3*S] = -1
    return J  
#------------------------------------------------------------------------------   
def plotresult(ti,yi,xkbest):
    import matplotlib.pyplot as plt   
    S = int((len(xkbest)-1)/3)
    T = np.arange(min(ti),max(ti),0.1)
    yhat = hyperbolicIO(T,xkbest)
    print(xkbest)
    plt.scatter(ti, yi, color='darkred', marker='x')
    plt.plot(T, yhat, color='green', linestyle='solid', linewidth = 1)
    plt.xlabel('Resistance (kOhm)')
    plt.ylabel('Voltage Gain')
    plt.title(str(S)+'-Node Hyperbolic Model | FV:'+str(fvalidationBest),fontstyle='italic')
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
    plt.legend(['hyperbolic model','collected data'])
    plt.show()     
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 
trainingindices = np.arange(0,len(ti),2)    
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)    
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
NODEmax = int((len(traininginput)-1)/3)
#------------------------------------------------------------------------------    
NODE = []; FV =  []; globalbest = 1e10; Sbest = 0;
for S in range(2,NODEmax):
    xk = np.random.random(3*S+1)-0.5
    k = 0; C1 = True; C2 = True; C3 = True; C4 = True; fvalidationBest = 1e99; kbest = 0
    ek = error(xk,traininginput,trainingoutput)
    ftraining = sum(ek**2)
    FTRA = [ftraining]
    evalidation = error(xk,validationinput,validationoutput)
    fvalidation = sum(evalidation**2)
    FVAL = [fvalidation]
    ITERATION = [k]
    # print('k:',k,' f',format(ftraining,'f'))
    mu = 1; muscal = 10; I = np.identity(3*S+1)
    while C1 & C2 & C3 & C4:
        ek = error(xk,traininginput,trainingoutput)
        Jk = findJacobian(traininginput,xk)
        gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])  
        Hk = 2*Jk.transpose().dot(Jk) + 1e-8*I
        ftraining = sum(ek**2)
        sk = 1
        loop = True
        while loop:
            zk = -np.linalg.inv(Hk+mu*I).dot(gk)
            zk = np.array(zk.tolist()[0])
            ez = error(xk+sk*zk,traininginput,trainingoutput)
            fz = sum(ez**2)
            if fz<ftraining:
                pk = 1*zk
                mu = mu/muscal
                k += 1
                xk = xk + sk*pk
                loop = False
                # print('k:',k,' f',format(fz,'f'))
            else:
                mu = mu*muscal
                if mu>mumax:
                    loop = False
                    C2   = False
        evalidation = error(xk,validationinput,validationoutput)
        fvalidation = sum(evalidation**2)
        if fvalidation<fvalidationBest:
            fvalidationBest = 1*fvalidation
            xkbest = 1*xk
            kbest = k
        FTRA.append(ftraining)
        FVAL.append(fvalidation)
        ITERATION.append(k)
        # --------------------
        C1 = k<MaxIter
        C2 = epsilon1<abs(ftraining-fz)
        C3 = epsilon2<np.linalg.norm(sk*pk)
        C4 = epsilon3<np.linalg.norm(gk)
        # --------------------    
    #------------------------------------------------------------------------------

    plotresult(ti,yi,xkbest)
    NODE.append(S)
    FV.append(fvalidationBest)
    if fvalidationBest<globalbest:
        globalbest = 1*fvalidationBest
        Sbest = S
    print('Node Count:', S, ' Fvalbest: ', fvalidationBest, ' GlobalFvalbest:', globalbest)
#------------------------------------------------------------------------------    
import matplotlib.pyplot as plt    
plt.bar(NODE, FV, color='orange', width = 0.4, linestyle='solid', linewidth = 1)
plt.bar(Sbest, globalbest, color='blue', width = 0.4, linestyle='solid', linewidth = 1)
plt.xlabel('Node count')
plt.ylabel('validation performance')
plt.title('Hyperbolic Model Validation Performances',fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
plt.show()    
#------------------------------------------------------------------------------ 
   
