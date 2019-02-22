import numpy as np
import RigidBodyKinematics as RBK






def epsilon_ki(normal,transform,S_i,D_i):



	x_S = transform[0:3]
	mrp_L0_S_bar = transform[3:6]

	x_D = transform[6:9]
	mrp_L0_D_bar = transform[9:12]

	L0_D_bar = RBK.mrp_to_dcm(mrp_L0_D_bar)
	L0_S_bar = RBK.mrp_to_dcm(mrp_L0_S_bar)


	return np.inner( L0_D_bar @ normal,L0_S_bar @ S_i + x_S - L0_D_bar @ D_i - x_D)



def partial_epsilon_ki_partial_transforms(normal,transform,S_i,D_i):


	x_S = transform[0:3]
	mrp_L0_S_bar = transform[3:6]

	x_D = transform[6:9]
	mrp_L0_D_bar = transform[9:12]

	L0_D_bar = RBK.mrp_to_dcm(mrp_L0_D_bar)
	L0_S_bar = RBK.mrp_to_dcm(mrp_L0_S_bar)

	partial = np.zeros(12)

	partial[0:3] = L0_D_bar @ normal
	partial[3:6] = - 4 * RBK.tilde(S_i) @ L0_S_bar.T @ L0_D_bar @ normal

	partial[6:9] = - L0_D_bar @ normal
	partial[9:12] = 4 * RBK.tilde(D_i) @ normal -  4 * RBK.tilde(normal) @ L0_D_bar.T @ (L0_S_bar @ S_i + x_S - L0_D_bar @ D_i - x_D)

	return partial

normal = np.array([1,2,3])
normal = normal / np.linalg.norm(normal) 

mrp_L0_D_bar = np.array([-0.2,0.1,0.2])
mrp_L0_S_bar = np.array([0.2,-0.1,0.1])

x_S = np.array([1,3,4])
x_D = np.array([-1,-3,2])

S_i = np.array([-1,-3,4])
D_i = np.array([-3,-3,1])

transform_bar = np.array(list(x_S) + list(mrp_L0_S_bar) + list(x_D) + list(mrp_L0_D_bar))
print(transform_bar)
e_bar = epsilon_ki(normal,transform_bar,S_i,D_i)
partial = partial_epsilon_ki_partial_transforms(normal,transform_bar,S_i,D_i)

dtransform = 1e-2 * transform_bar

e = epsilon_ki(normal,dtransform + transform_bar,S_i,D_i)

de = e - e_bar

print(de)
print(partial @ dtransform)
print(abs(de - partial @ dtransform) / (partial @ dtransform) * 100)




















