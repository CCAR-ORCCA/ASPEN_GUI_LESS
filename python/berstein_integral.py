import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy

def partial_bernstein(u,v,i,j,n) :

	partials = np.zeros(2)

	partials[0] = n *( bernstein(u, v,i - 1,j,n - 1) - bernstein(u, v,i,j,n - 1))
	partials[1] = n *( bernstein(u, v,i,j - 1,n - 1) - bernstein(u, v,i,j,n - 1))

	return partials
def dBdu(u,v,i,j,n) :

	
	return n *( bernstein(u, v,i - 1,j,n - 1) - bernstein(u, v,i,j,n - 1))
def dBdv(u,v,i,j,n) :

	return n *( bernstein(u, v,i ,j - 1 ,n - 1) - bernstein(u, v,i,j,n - 1))
def bernstein(u, v,i,j,n):

	if (i < 0 or i > n or j < 0 or j > n or i + j > n):
		return 0
	
	if (n == 0):
		return 1
	
	coef =  np.math.factorial(n) / (
		np.math.factorial(i)
		* np.math.factorial(j) 
		* np.math.factorial(n - i - j));

	return coef * (u ** i *  v ** j * (1 - u - v) ** ( n - i - j ))
def vol_fun(u,v,i,j,k,n):

	return (bernstein(u,v,i[0],i[1],n) 
		* dBdu(u,v,j[0],j[1],n) 
		* dBdv(u,v,k[0],k[1],n))/3


def cm_fun(u,v,i,j,k,l,n):

	return (bernstein(u,v,i[0],i[1],n)
	* bernstein(u,v,j[0],j[1],n) 
		* dBdu(u,v,k[0],k[1],n) 
		* dBdv(u,v,l[0],l[1],n))/2

def gfun(x):
	return 0

def hfun(x):
	return 1 - x

def generate_vol_int_table(degree):
	
	N_c = (degree + 1) * (degree + 2)/2
	indices = []

	for i in range(degree + 1):
	    for j in range(degree + 1 - i):
	        indices += [[i,j,degree - i - j]]

	results = []

	for i in range(len(indices)):

		i_i = indices[i][0]
		i_j = indices[i][1]


		for j in range(len(indices)):

			j_i = indices[j][0]
			j_j = indices[j][1]

			for k in range(len(indices)):

				k_i = indices[k][0]
				k_j = indices[k][1]

				indices_i = [i_i,i_j]
				indices_j = [j_i,j_j]
				indices_k = [k_i,k_j]


				args = indices_i, indices_j,indices_k , degree

				integral = integrate.dblquad(vol_fun,0,1,gfun,hfun,args = args)
				
				results += [[int(i_i),int(i_j),int(j_i),int(j_j),int(k_i),int(k_j),integral[0]]]


	results = np.vstack(results)
	print "Size: " 
	print results.shape

	np.savetxt("volume_integral_degree_" + str(degree)+ ".txt",results,delimiter = ",",newline = ",\n")



def generate_cm_int_table(degree):
	
	N_c = (degree + 1) * (degree + 2)/2
	indices = []

	for i in range(degree + 1):
	    for j in range(degree + 1 - i):
	        indices += [[i,j,degree - i - j]]

	results = []

	for i in range(len(indices)):

		i_i = indices[i][0]
		i_j = indices[i][1]


		for j in range(len(indices)):

			j_i = indices[j][0]
			j_j = indices[j][1]

			for k in range(len(indices)):

				k_i = indices[k][0]
				k_j = indices[k][1]

				for l in range(len(indices)):

					l_i = indices[l][0]
					l_j = indices[l][1]

					indices_i = [i_i,i_j]
					indices_j = [j_i,j_j]
					indices_k = [k_i,k_j]
					indices_l = [l_i,l_j]



					args = indices_i, indices_j,indices_k,indices_l , degree

					integral = integrate.dblquad(cm_fun,0,1,gfun,hfun,args = args)
					
					results += [[int(i_i),int(i_j),int(j_i),int(j_j),int(k_i),int(k_j),int(l_i),int(l_j),integral[0]]]


	results = np.vstack(results)
	print "Size: " 
	print results.shape

	np.savetxt("cm_integral_degree_" + str(degree)+ ".txt",results,delimiter = ",",newline = ",\n")
generate_cm_int_table(4)
