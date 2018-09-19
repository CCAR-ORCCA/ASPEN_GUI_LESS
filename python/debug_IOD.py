import numpy as np
import RigidBodyKinematics as RBK


def X_prime_k(NL,M_km1,X_k,X_km1):
	return NL.dot(M_km1.T).dot(X_k - X_km1)

def dX_prime_k(NL,M_km1,X_k,X_km1,d_X_k,d_X_km1,dmrp_M_km1):

	a = M_km1.T.dot(X_k - X_km1)
	U = -4 * RBK.tilde(a)
	return NL.dot(M_km1.T.dot(d_X_k - d_X_km1) + U.dot(dmrp_M_km1))


mrp_NL = np.array([0.1,0.3,0.5])
NL = RBK.mrp_to_dcm(mrp_NL)


mrp_M_km1 = np.array([-0.1,-0.3,0.5])
M_km1 = RBK.mrp_to_dcm(mrp_M_km1)

X_k = np.random.randn(3)
X_km1 = np.random.randn(3)


dmrp_M_km1 = 0.001 * mrp_M_km1
M_km1_bar_bar = RBK.mrp_to_dcm(mrp_M_km1 + dmrp_M_km1)
d_X_k = 0.001 * X_k
d_X_km1 = 0.001 * X_km1

X_bar = X_prime_k(NL,
	M_km1,
	X_k,
	X_km1)

X_bar_bar = X_prime_k(NL,
	M_km1_bar_bar,
	X_k + d_X_k,
	X_km1 + d_X_km1)
dX = dX_prime_k(NL,
	M_km1,
	X_k,
	X_km1,
	d_X_k,
	d_X_km1,
	dmrp_M_km1)


print (X_bar_bar - X_bar - dX)/dX * 100

