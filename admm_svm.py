# coding: utf-8
from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from multiprocessing import Pool

""" setup config data """
""" gaussian covariance and mean """
cov = [[60,0],[0,60]]
mean1 = [10,10]
mean2 = [-10,-10]
""" data dimensionality """
n = 2;
""" number of examples """
m = 1000;
""" number of blocks """
b = 10;
""" TRAINING SET """
""" positive examples """
M = m/2;
x1,y1 = random.multivariate_normal(mean1,cov,M).T
""" negative examples """
x2,y2 = random.multivariate_normal(mean2,cov,M).T
""" Generate 20% data for TEST SET """
M_test = m/10;
""" positive examples """
x1_test,y1_test = random.multivariate_normal(mean1,cov,M_test).T
""" negative examples """
x2_test,y2_test = random.multivariate_normal(mean2,cov,M_test).T
X_test = vstack([hstack([x1_test,x2_test]),hstack([y1_test,y2_test])])
Y_test = hstack([ones((1,M_test)),-ones((1,M_test))])
""" add x0 to X and change Y for easy handling """
X_test = vstack([ones((1,2*M_test)),X_test])
Y_test = dot(ones((n+1,1)),Y_test)

""" merge into one X(2,m) training set and Y(1,m) first half
positive examples second half negative examples """
""" merge x1 into x2, y1 into y2 and then into X """
X = vstack([hstack([x1,x2]),hstack([y1,y2])])
Y = hstack([ones((1,M)),-ones((1,M))])
""" show the sizes """
print X.shape
print Y.shape

""" plot the training examples for the two classes """
plt.plot(X[0,0:M],X[1,0:M],'x'); plt.plot(X[0,-M:],X[1,-M:],'r+'); plt.show()
""" add x0 to X  """
X = vstack([ones((1,m)),X])
""" partition the data into uneven blocks """
p = sort(random.randint(0,b,m))
N = max(p)+1
X_dict = {}
Y_dict = {}
for i in range(N):
    X_dict[i] = X[:,p==i]
    Y_dict[i] = dot(ones((n+1,1)),Y[:,p==i])
""" create the initial guess and add bias term """
w = zeros((n+1,N))
z = zeros((n+1,N))
u = zeros((n+1,N))
def l2_loss(w, X, y):
#    p = dot(w.T, X)
#    p[p<0] = -1
#    p[p>=0] = 1
#    return sum(p!=y[0])
    A = maximum(1 - dot(w.T,y *X),0)**2
    return sum(A);

def objective2(w, X, y, z, u, C, rho):
    """ if w arrives as an array w.shape == (n+1,) then we change it to a matrix of shape (n+1,1) """
    if w.shape == (n+1,):
        w = array([w]).T
    A = maximum(1 - dot(w.T,y *X),0)
    func = (C * sum(A**2)) + ((rho/2) * sum((w - z + u)**2))
    """ compute the gradient (rho + C * sum(x*x')*w - 2*C*sum(y*x) - rho * (z+u)"""
    """ restrict only to examples where (1-y*w'*x)>0) """
    X_pos = X[:,nonzero(A>0)[0]]
    Y_pos = Y[:,nonzero(A>0)[0]]
    grad = dot((rho + (2 * C * dot(X_pos,X_pos.T))),w) - (2*C*sum(Y_pos * X_pos,1,out=zeros((n+1,1)))) - (rho * ( z + u ))
    """ return the function value and the gradient """
    return [func, grad.ravel()]

def parallel_optim(w, X, y, z, u, C, rho):
    """ optimization that happens at each process """
    min_result = fmin_l_bfgs_b(objective2, w, args= [X,y, z, u, C, rho])
    return min_result[0] 

def f(x,wave):
    """ used to plot the hyperplane in 2 dimensions, x is an array """
    y = ((-wave[1]/wave[2])*x) - (wave[0]/wave[2])
    return y

MAX_ITER = b;
C = 1.0
rho = 1.0
plotme = []
loss_plot = []
""" define a pool of 4 processes"""
pool = Pool(5)
results = []
t = arange(-20.0,20.0,0.1)
beta = 1.5
""" even if this works in parallel this version runs the updates in serial """
for k in range(MAX_ITER):
    """ at every iteration we scatter the data to the nodes where w is minimised, on return we update w, z, u """
    for i in range(N):
        wi = w[:,i:i+1]
        zi = z[:,i:i+1]
        ui = u[:,i:i+1]
        Xi = X_dict[i]
        Yi = Y_dict[i]
        wmin_future = pool.apply_async(parallel_optim,[wi, Xi, Yi, zi, ui, C, rho])
        results.append(wmin_future)
    """ return check the async results for each of the local minimizations """
    for i in range(N):
        wmin = results[i].get()
        w[:,i:i+1] = array([wmin]).T
    """ over relaxation """
    w_hat = (beta * w) + ((1-beta)*z)
    """ cleanup the results """
    del results[:]
    """ update z with the mean of w + u taking rho into account """
    z = sum(w_hat + u,1,out=zeros((n+1,1))) / (N + (1/rho))
    z = dot(z,ones((1,N)))
    """ update u """
    u = u + (w_hat - z)
    wave = mean(w,1)
    print wave
    plotme.append(f(t,wave))
    """ evaluate the cost function on the test set """
    loss_plot.append(l2_loss(wave,X_test, Y_test))
plt.figure(1)
plt.subplot(121)
plt.plot(X[1,0:M],X[2,0:M],'x'); plt.plot(X[1,-M:],X[2,-M:],'r+');
for t_plot in plotme:
    plt.plot(t,t_plot);
plt.figure(1)
plt.subplot(122)
plt.plot(arange(MAX_ITER),loss_plot);plt.show()
plt.show();

