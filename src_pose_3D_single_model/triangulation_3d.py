from scipy.optimize import least_squares
import scipy.io as sio
import torch

proj_params = sio.loadmat('proj_params_101019_corrected_new.mat')
proj_params = proj_params['proj_params']
proj_params = proj_params[None, :, :]

def triangulation_3d(pose_b, pose_s1, pose_s2, proj_params):
    coor = torch.zeros(2,3)
    coor[:,0] = pose_b.T
    coor[:,1] = pose_s1.T
    coor[:,2] = pose_s2.T

    fa1p00 = proj_params[0,0,0]
    fa1p10 = proj_params[0,0,1]
    fa1p01 = proj_params[0,0,2]
    fa1p20 = proj_params[0,0,3]
    fa1p11 = proj_params[0,0,4]
    fa1p30 = proj_params[0,0,5]
    fa1p21 = proj_params[0,0,6]
    fa2p00 = proj_params[0,1,0]
    fa2p10 = proj_params[0,1,1]
    fa2p01 = proj_params[0,1,2]
    fa2p20 = proj_params[0,1,3]
    fa2p11 = proj_params[0,1,4]
    fa2p30 = proj_params[0,1,5]
    fa2p21 = proj_params[0,1,6]
    fb1p00 = proj_params[0,2,0]
    fb1p10 = proj_params[0,2,1]
    fb1p01 = proj_params[0,2,2]
    fb1p20 = proj_params[0,2,3]
    fb1p11 = proj_params[0,2,4]
    fb1p30 = proj_params[0,2,5]
    fb1p21 = proj_params[0,2,6]
    fb2p00 = proj_params[0,3,0]
    fb2p10 = proj_params[0,3,1]
    fb2p01 = proj_params[0,3,2]
    fb2p20 = proj_params[0,3,3]
    fb2p11 = proj_params[0,3,4]
    fb2p30 = proj_params[0,3,5]
    fb2p21 = proj_params[0,3,6]
    fc1p00 = proj_params[0,4,0]
    fc1p10 = proj_params[0,4,1]
    fc1p01 = proj_params[0,4,2]
    fc1p20 = proj_params[0,4,3]
    fc1p11 = proj_params[0,4,4]
    fc1p30 = proj_params[0,4,5]
    fc1p21 = proj_params[0,4,6]
    fc2p00 = proj_params[0,5,0]
    fc2p10 = proj_params[0,5,1]
    fc2p01 = proj_params[0,5,2]
    fc2p20 = proj_params[0,5,3]
    fc2p11 = proj_params[0,5,4]
    fc2p30 = proj_params[0,5,5]
    fc2p21 = proj_params[0,5,6]
    
    fun = lambda x: ((fa1p00+fa1p10*x[2]+fa1p01*x[0]+fa1p20*x[2]**2+fa1p11*x[2]*x[0]+fa1p30*x[2]**3+fa1p21*x[2]**2*x[0] - coor[0,0])**2 + 
            (fa2p00+fa2p10*x[2]+fa2p01*x[1]+fa2p20*x[2]**2+fa2p11*x[2]*x[1]+fa2p30*x[2]**3+fa2p21*x[2]**2*x[1] - coor[1,0])**2 + 
            (fb1p00+fb1p10*x[0]+fb1p01*x[1]+fb1p20*x[0]**2+fb1p11*x[0]*x[1]+fb1p30*x[0]**3+fb1p21*x[0]**2*x[1] - coor[0,1])**2 + 
            (fb2p00+fb2p10*x[0]+fb2p01*x[2]+fb2p20*x[0]**2+fb2p11*x[0]*x[2]+fb2p30*x[0]**3+fb2p21*x[0]**2*x[2] - coor[1,1])**2 + 
            (fc1p00+fc1p10*x[1]+fc1p01*x[0]+fc1p20*x[1]**2+fc1p11*x[1]*x[0]+fc1p30*x[1]**3+fc1p21*x[1]**2*x[0] - coor[0,2])**2 + 
            (fc2p00+fc2p10*x[1]+fc2p01*x[2]+fc2p20*x[1]**2+fc2p11*x[1]*x[2]+fc2p30*x[1]**3+fc2p21*x[1]**2*x[2] - coor[1,2])**2) 

    bnds = ((-30,-30,50), (30,30,100))
    res = least_squares(fun, (0, 0, 70), method='dogbox', bounds = bnds)
    print(fun([0, 0, 70]))
    return res.x, res.fun

 
pose_b = torch.tensor([568.0852,  481.2964])
pose_s1 = torch.tensor([149.9895,  126.1637])
pose_s2 = torch.tensor([505.1766,  128.2617])

x, fun = triangulation_3d(pose_b, pose_s1, pose_s2, proj_params)
print(x, fun)



