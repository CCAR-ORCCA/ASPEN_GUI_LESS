# MIT License

# Copyright (c) 2018 Benjamin Bercovici

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def prv_to_quat(axis,angle):
    '''
    Principal rotation vector to Quaternion
    Inputs:
    -------
    - axis : (3-by-1 np.array) Rotation axis (does not have to be normalized)
    - angle : (scalar) Rotation angle (radians)
    Outputs:
    ------
    - q : (4-by-1 np.array) quaternion
    '''

    q = np.zeros(4)
    quat[0] = np.cos(angle/2)
    quat[1:] = np.sin(angle/2) * axis/np.linalg.norm(axis)
    return q



def dcm_to_prv(dcm):
    '''
    Returns the principal rotation vector corresponding to this dcm
    always return the "short" rotation
    Inputs:
    -------
    dcm : (3-by-3 np.array) dcm
    Outputs:
    -------
    prv : (3-by-1 np.array) prv
    '''

    cos_phi = 0.5 * (np.trace(dcm) - 1)
    if (cos_phi - 1 > 0):
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        e_dir = 1./(2 * sin_phi) * np.array([dcm[1,2] - dcm[2,1],dcm[2,0] - dcm[0,2],dcm[0,1] - dcm[1,0]])

        return np.arccos(cos_phi) * e_dir
    else:
        return np.zeros(3)



def mrp_to_prv(mrp):
    '''
    Returns the principal rotation vector corresponding to this mrp
    always return the "short" rotation
    Inputs:
    -------
    mrp : (3-by-1 np.array) mrp
    Outputs:
    -------
    prv : (3-by-1 np.array) prv
    '''

    return dcm_to_prv(mrp_to_dcm(mrp))

def quat_to_mrp(quat):
    '''
    Quaternion to MRP
    Inputs:
    ------
    - quat : (4-by-1 np.array) quaternion
    Outputs:
    -------
    - sigma : (3-by-1 np.array) MRP
    '''

    sigma = quat[1:] / (1 + quat[0])

    if(np.linalg.norm(sigma) > 1):
        sigma = - sigma /  np.linalg.norm(sigma) ** 2
    return sigma


def mrp_to_quat(mrp):
    '''
    Converts a mrp to the corresponding short quaternion
    Inputs:
    ------
    - mrp : (3-by-1 np.array) MRP
    Outputs:
    -------
    - quat : (4-by-1 np.array) quaternion
    '''
    return dcm_to_quat(mrp_to_dcm(mrp))



def shadow_mrp(mrp, force_switch = False) :
    '''
    Switches MRP to its shadow
    Inputs:
    ------
    - mrp : (3-by-1 np.array) MRP
    - force_switch: (bool) true is the mrp must be switched
    Outputs:
    -------
    - mrp : (3-by-1 np.array) MRP that may have switched 
    '''
    if (np.linalg.norm(mrp) > 1 or force_switch is True) :
        return - mrp / np.inner(mrp, mrp)
    
    else :
        return mrp
    






def mrp_to_dcm(sigma):
    '''
    MRP to dcm
    Inputs:
    ------
    - sigma : (3-by-1) MRP
    Outputs:
    ------
    - dcm : (3-by-3 ) dcm
    '''
    dcm = np.eye(3) + (8 * tilde(sigma).dot(tilde(sigma)) - 4 * (1 - np.linalg.norm(sigma) ** 2) * tilde(sigma))/(1 + np.linalg.norm(sigma)**2)**2
    return dcm

def dcm_to_321(dcm):
    '''
    dcm to 321 Euler angles
    Inputs :
    --------
    dcm : (3-by-3) dcm
    Outputs:
    --------
    - euler_angles : set of 321 euler angles (yaw,pitch,roll)
    '''

    euler_angles = np.zeros(3)
    euler_angles[0] = np.arctan2(dcm[0,1],dcm[0,0])
    euler_angles[1] = - np.arcsin(dcm[0,2])
    euler_angles[2] = np.arctan2(dcm[1,2],dcm[2,2])
    return euler_angles



def tilde(a):

    '''
    Returns the skew-symmetric matrix corresponding to the linear mapping cross(a,.)
    Inputs : 
    ------
    a : (3-by-1 np.array) vector
    Outputs:
    ------
    atilde : (3-by-3 np.array) linear mapping matrix
    '''
    atilde = np.array([[0,-a[2],a[1]],
        [a[2],0,-a[0]],
        [-a[1],a[0],0]])
    return atilde

def add_MRP(sigma_p,sigma_pp):
    '''
    Adds two MRPS together, as in [FN](sigma) = [FB](sigma_pp) [BN](sigma_p) 
    Inputs:
    -------
        - sigma_p : (3-by-1 vector) first MRP (as in "rightmost" rotation matrix argument)
        - sigma_pp : (3-by-1 vector) second MRP
    Outputs:
    -------
        - sigma : (3-by-1 vector) MRP corresponding to the addition of the two rotations 
    '''
    sig_p = np.linalg.norm(sigma_p)
    sig_pp = np.linalg.norm(sigma_pp)
    sigma =  ( (1 - sig_p ** 2) * sigma_pp + (1 - sig_pp ** 2) * sigma_p - 2 * np.cross(sigma_pp,sigma_p) ) / ( 1 + (sig_p * sig_pp) ** 2 - 2 * np.inner(sigma_p,sigma_pp))
    return sigma

def M1(t):
    '''
    Evaluates the dcm matrix describing a rotation about the first axis of the current reference frame
    Inputs:
    ------
        - t : (scalar) angle in radians
    Outputs:
    ------
        - M : (3-by-3 np.array) orthonormal rotation matrix
    '''

    M = np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
    return M

def M2(t):
    '''
    Evaluates the dcm matrix describing a rotation about the second axis of the current reference frame
    Inputs:
    ------
        - t : (scalar) angle in radians
    Outputs:
    ------
        - M : (3-by-3 np.array) orthonormal rotation matrix
    '''

    M = np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
    return M

def M3(t):
    '''
    Evaluates the dcm matrix describing a rotation about the third axis of the current reference frame
    Inputs:
    ------
        - t : (scalar) angle in radians
    Outputs:
    ------
        - M : (3-by-3 np.array) orthonormal rotation matrix
    '''

    M = np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])
    return M




def dcm_to_mrp(dcm) :
    '''
    Converts the provided dcm to mrp
    '''
    return quat_to_mrp(dcm_to_quat(dcm))



def euler313_to_dcm(theta_1,theta_2,theta_3):
    '''
    Computes the dcm corresponding to the 3-1-3 rotation sequence 
    Inputs:
    ------
        - theta_1 : (scalar) first angle of the rotation sequence (aka right ascension)
        - theta_2 : (scalar) second angle of the rotation sequence (aka inclination)
        - theta_3 : (scalar) third angle of the rotation sequence (aka argument of perigee + true anomaly)
    Outputs:
    ------
        - M : (3-by-3 np.array) orthonormal rotation matrix
    '''

    M = M3(theta_3).dot(M1(theta_2)).dot(M3(theta_1))
    return M

def euler321_to_dcm(theta_1,theta_2,theta_3):
    '''
    Computes the dcm corresponding to the 3-1-3 rotation sequence 
    Inputs:
    ------
        - theta_1 : (scalar) first angle of the rotation sequence (aka yaw)
        - theta_2 : (scalar) second angle of the rotation sequence (aka pitch)
        - theta_3 : (scalar) third angle of the rotation sequence (aka roll)
    Outputs:
    ------
        - M : (3-by-3 np.array) orthonormal rotation matrix
    '''
    
    M = M1(theta_3).dot(M2(theta_2)).dot(M3(theta_1))
    return M



def dcm_to_quat(dcm) :
    '''
    Computes the short quaternion corresponding to the provided dcm
    Inputs:
    ------
        - dcm: (3-by-3 np.array) orthonormal rotation matrix
    Outputs:
    ------
        - quat : (4-by-1 array) quaternion
    '''

    quat = np.zeros(4)
    quat_s = np.array([
        1. / 4. * ( 1 + np.trace(dcm) ),
        1. / 4. * (1 + 2 * dcm[0, 0] - np.trace(dcm)),
        1. / 4. * (1 + 2 * dcm[1, 1] - np.trace(dcm)),
        1. / 4. * (1 + 2 * dcm[2, 2] - np.trace(dcm))])


    max_coef_index = np.argmax(quat_s)
    quat[max_coef_index] = np.sqrt(np.amax(quat_s))

    if max_coef_index == 0:
        quat[1] = 1. / 4. * (dcm[1, 2] - dcm[2, 1]) / quat[0]
        quat[2] = 1. / 4. * (dcm[2, 0] - dcm[0, 2]) / quat[0]
        quat[3] = 1. / 4. * (dcm[0, 1] - dcm[1, 0]) / quat[0]

    elif max_coef_index == 1:
        quat[0] = 1. / 4. * (dcm[1, 2] - dcm[2, 1]) / quat[1]
        quat[2] = 1. / 4. * (dcm[0, 1] + dcm[1, 0]) / quat[1]
        quat[3] = 1. / 4. * (dcm[2, 0] + dcm[0, 2]) / quat[1]

    elif max_coef_index == 2:
        quat[0] = 1. / 4. * (dcm[2, 0] - dcm[0, 2]) / quat[2]
        quat[1] = 1. / 4. * (dcm[0, 1] + dcm[1, 0]) / quat[2]
        quat[3] = 1. / 4. * (dcm[1, 2] + dcm[2, 1]) / quat[2]


    else:
        quat[0] = 1. / 4. * (dcm[0, 1] - dcm[1, 0]) / quat[3]
        quat[1] = 1. / 4. * (dcm[2, 0] + dcm[0, 2]) / quat[3]
        quat[2] = 1. / 4. * (dcm[1, 2] + dcm[2, 1]) / quat[3]


    return quat



