import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle


#  Question 6
def q6c():

    a = np.array([1., 2., 2., 4.])
    Ha = np.array([-5., 0., 0., 0.])
    a = np.reshape(a, (4,1))
    Ha = np.reshape(Ha, (4,1))
    v = a - Ha
    H = np.eye(4) - 2 * np.dot(v, v.T)/np.dot(v.T, v)
    print('\nH: ')
    print (H)
    print('\nHa: ')
    print(Ha)


# Question 8

def fitpoly():

    with open('data2.pickle', 'rb') as f:
        date = pickle.load(f)

    [t, y] = data
    d = 5
    M = np.size(t)
    A = np.ones(M, d + 1)

    for i in range (1, d + 1):
        A[:, i] = A[:, i - 1] * t

    b = y

# part b

    B = np.dot(A.T, A)
    c = np.dot(A.T, b)
    U = la.cholesky(B)
    z = la.solve_triangular(U.T, c, lower = True)
    x = la.solve_triangular(U, z)
    print ('\n')
    print (' Question 8(b): ')
    print ('\n x = %s' % (np.array_str(x)))

# part c

    N = 1000
    t = np.linspace(1, 5, N)
    A = np.ones((N, d + 1))

    for i in range(1, d + 1):
        A[:, i] = A[:, i -1] * t

    y = np.dot(A, x)
    plt.plot(t, y, 'g')
    [t, y] = data
    plt.plot(t, y, 'b.')
    plt.suptitle('Question 8(c). Data and fitted polynomial')
    plt.xlabel('t')
    plt.ylabel('y')


# Question 9

def fitNormal(X):
    (M, N) = X.shape
    mu = X.sum(axis = 0)/M
    Xc = X - mu
    Sigma = np.dot(Xc.T, Xc)/M
    return (mu, Sigma)

# part c

def MVNinv(X, mu, Sigma):
    [M, N] = X.shape
    P = np.zeros(M)
    SigmaInv = la.inv(Sigma)
    alpha = np.sqrt((2 * np.pi) ** N) * (la.det(Sigma))

    for m in range (0,M):
        x = X[m, :] - mu
        z = np.dot(x, np.dot(SigmaInv, x.T))
        P[m] = np.exp(-z/2)/alpha

    return(P)

# part d

def MVNchol(X, mu, Sigma):
    [M, N] = X.shape
    P = np.zeros(M)
    L = la.cholesky(Sigma, lower = True)
    alpha = np.sqrt((2 * np.pi) ** N) * (la.det(Sigma))

    for m in range (0, M):
        x = X[m, :] - mu
        y = la.solve_triangular(L, x, lower = True, check_finite = False)
        z = np.dot(y,y)
        P[m] = np.exp(-z/2)/alpha

    return(P)


# Question 10

def testOCR():

# part a
    
    mu = [2, 3]
    Sigma = [[4, 3], [3, 4]]
    N = 1000
    X = rnd.multivariate_normal(mu, Sigma, N)
    plt.plot(X[:,0], X[:,1],'.')
    plt.suptitle('Question 10(a). 1000 points of multirvariate normal data')

# part b
    
    print('\n \nQuestion 10(b): ')
    for N in [10, 100, 10000, 1000000]:
        X = rnd.multivariate_normal(mu, Sigma, N)
        [mu0, Sigma0] = fitNormal(X)
        print(' ')
        print('N = %d'%(N))
        print ('mu = ')
        print (mu0)
        print ('Sigma = ')
        print(Sigma0)

# part c

    print ('\n\nQuestion 10(c): ')
    N = 100
    X = rnd.multivariate_normal(mu, Sigma, N)
    p1 = MVNinv(X, mu, Sigma)
    p2 = MVNchol(X, mu, Sigma)
    print('norm of p1-p2 = % .4g'%(la.norm(p1-p2)))













    
