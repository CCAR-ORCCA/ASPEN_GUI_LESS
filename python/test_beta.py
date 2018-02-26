import numpy as np
from scipy.integrate import quad
from scipy.misc import comb



def u1(u,i,j,k,l,n):
	return u ** (i + j + k + l - 1) * (1 - u) ** (4 * n - i - j - k - l)


def u2(u,i,j,k,l,n):
	return u ** (i + j + k + l) * (1 - u) ** (4 * n - i - j - k - l - 1)

def u3(u,a,b):
	return u ** a * (1 - u) ** (b - a)

def beta(i,j,k,l,n):

	args = i,j,k,l,n

	return comb(n,i) * comb(n,j) * comb(n,k) * n * (
	 	comb(n - 1,l - 1) * Sba_b(i + j + k + l - 1,4 * n - i - j - k - l - 1)
	 	- comb(n - 1 , l ) * Sba_b(i + j + k + l, 4 * n - i - j - k - l - 1))


def Sba_b(a,b):
	if a < 0 or b - a < 0:
		return 0
	else:
		return sum([comb(b - a,k) * (-1) ** k  / (b - k + 1) for k in range(b -a + 1)])



print beta(1,1,1,1,1)
print beta(0,0,0,1,1)
print beta(0,0,1,0,1)
print beta(0,0,1,1,1)
print beta(0,1,0,0,1)
print beta(0,1,0,1,1)
print beta(0,1,1,0,1)
print beta(0,1,1,1,1)
print beta(1,0,0,0,1)
print beta(1,0,0,1,1)

print beta(1,0,1,0,1)
print beta(1,0,1,1,1)
print beta(1,1,0,0,1)
print beta(1,1,0,1,1)
print beta(1,1,1,0,1)
print beta(1,1,1,1,1)






