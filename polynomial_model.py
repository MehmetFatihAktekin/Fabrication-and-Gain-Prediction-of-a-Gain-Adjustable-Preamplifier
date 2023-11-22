import numpy as np
import math
from dataset import ti,yi
#------------------------------------------------------------------------------
def polinomIO(t,x):
    yhat = []
    for ti in t:
        toplam = 0
        for i in range(0,len(x)):
            toplam += x[i]*ti**i
        yhat.append(toplam)
    return yhat
#------------------------------------------------------------------------------
def findx(ti,yi,polinomderecesi):
    numofdata = len(ti)    
    J = -np.ones((numofdata,1))
    for n in range(1,polinomderecesi+1):
        J = np.hstack((J,-np.ones((numofdata,1))*np.array(ti).reshape(numofdata,1)**n))
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x  
#------------------------------------------------------------------------------    
def plotresult(ti,yi,x,fvalidation):
    import matplotlib.pyplot as plt
    print(x)
    
    T = np.arange(min(ti),max(ti),0.1)
    yhat = polinomIO(T,x)
    plt.scatter(ti, yi, color='darkred', marker='x')
    plt.plot(T, yhat, color='green', linestyle='solid', linewidth = 1)
    plt.xlabel('Resistance (kOhm)')
    plt.ylabel('Voltage Gain')
    plt.title(str(len(x)-1)+'.-th power polynomial model | FV:'+str(fvalidation),fontstyle='italic')
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
    plt.legend(['polynomial model','collected data'])
    plt.show()    
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 
trainingindices = np.arange(0,len(ti),2)    
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)    
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

fvalidationBest=10000
pdbest=0
PD = []; FV =  []
for polinomderecesi in range(2,11):
    x = findx(traininginput,trainingoutput,polinomderecesi)
    yhat = polinomIO(validationinput,x)
    e = np.array(validationoutput) - np.array(yhat)
    fvalidation = sum(e**2)
    
    if fvalidation<fvalidationBest:
        fvalidationBest=fvalidation
        pdbest=polinomderecesi
    
    PD.append(polinomderecesi)
    FV.append(fvalidation)
    print(polinomderecesi,fvalidation)
    plotresult(ti,yi,x,fvalidation)
#------------------------------------------------------------------------------    
import matplotlib.pyplot as plt    
plt.bar(PD, FV, color='orange', width = 0.4, linestyle='solid', linewidth = 1)

plt.bar(pdbest,fvalidationBest, color='blue', width = 0.4, linestyle='solid', linewidth = 1)

plt.xlabel('degree of the polynomial')
plt.ylabel('validation performance')
plt.title('Polynomial Model Validation Performances',fontstyle='italic')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
plt.show()    
#------------------------------------------------------------------------------ 
 

