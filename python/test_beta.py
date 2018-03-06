import numpy as np
from scipy.integrate import quad
from scipy.misc import comb



def u1(u,i,j,k,l,n):
	return u ** (i + j + k + l - 1) * (1 - u) ** (4 * n - i - j - k - l)


def u2(u,i,j,k,l,n):
	return u ** (i + j + k + l) * (1 - u) ** (4 * n - i - j - k - l - 1)

def u3(u,a,b):
	return u ** a * (1 - u) ** (b - a)


def u4(v,u,a,b):
	return v ** a * ( 1- u - v) ** b

def beta(i,j,k,l,n):

	args = i,j,k,l,n

	return comb(n,i) * comb(n,j) * comb(n,k) * n * (
	 	comb(n - 1,l - 1) * Sa_b(i + j + k + l - 1,4 * n - i - j - k - l - 1)
	 	- comb(n - 1 , l ) * Sa_b(i + j + k + l, 4 * n - i - j - k - l - 1))


def Sa_b(a,b):
	if a < 0 or b - a < 0:
		return 0
	else:
		return sum([comb(b - a,k) * (-1) ** k  / (b - k + 1) for k in range(b -a + 1)])



def Pba_b(a,b):
	if a < 0 or b < 0:
		return 0
	else:
		return sum([comb(b,k) * (-1) ** k  / (a + k + 1) for k in range(b + 1)])

a = 2
b = 3
u = 0.5
args = u,a,b

true_int = quad(u4,0,1 - u,args = args)
print true_int
print (1 - u) ** (1 + a + b) * Pba_b(a,b)



