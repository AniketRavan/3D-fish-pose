import torch
import argparse
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('-c','--coor_3d',default='coor_3d.mat', type=str, help='coor 3d path')
parser.add_argument('-p','--proj_params', default="proj_params_101019_corrected_new", type=str, help='path to calibrated camera parameters')
args = vars(parser.parse_args())

coor_3d_path = args['coor_3d']
coor_3d = sio.loadmat(coor_3d_path)
coor_3d = coor_3d['coor_3d']
proj_params_path = args['proj_params']
proj_params = sio.loadmat(proj_params_path)
proj_params = proj_params['proj_params']

coor_3d = torch.tensor(coor_3d)
proj_params = torch.tensor(proj_params)

def calc_proj_w_refra(coor_3d,proj_params):
    fa1p00 = proj_params[0,0]
    fa1p10 = proj_params[0,1]
    fa1p01 = proj_params[0,2]
    fa1p20 = proj_params[0,3]
    fa1p11 = proj_params[0,4]
    fa1p30 = proj_params[0,5]
    fa1p21 = proj_params[0,6]
    fa2p00 = proj_params[1,0]
    fa2p10 = proj_params[1,1]
    fa2p01 = proj_params[1,2]
    fa2p20 = proj_params[1,3]
    fa2p11 = proj_params[1,4]
    fa2p30 = proj_params[1,5]
    fa2p21 = proj_params[1,6]
    fb1p00 = proj_params[2,0]
    fb1p10 = proj_params[2,1]
    fb1p01 = proj_params[2,2]
    fb1p20 = proj_params[2,3]
    fb1p11 = proj_params[2,4]
    fb1p30 = proj_params[2,5]
    fb1p21 = proj_params[2,6]
    fb2p00 = proj_params[3,0]
    fb2p10 = proj_params[3,1]
    fb2p01 = proj_params[3,2]
    fb2p20 = proj_params[3,3]
    fb2p11 = proj_params[3,4]
    fb2p30 = proj_params[3,5]
    fb2p21 = proj_params[3,6]
    fc1p00 = proj_params[4,0]
    fc1p10 = proj_params[4,1]
    fc1p01 = proj_params[4,2]
    fc1p20 = proj_params[4,3]
    fc1p11 = proj_params[4,4]
    fc1p30 = proj_params[4,5]
    fc1p21 = proj_params[4,6]
    fc2p00 = proj_params[5,0]
    fc2p10 = proj_params[5,1]
    fc2p01 = proj_params[5,2]
    fc2p20 = proj_params[5,3]
    fc2p11 = proj_params[5,4]
    fc2p30 = proj_params[5,5]
    fc2p21 = proj_params[5,6]
    npts = coor_3d.shape[2]
    coor_b = torch.zeros(1,2,npts)
    coor_s1 = torch.zeros(1,2,npts)
    coor_s2 = torch.zeros(1,2,npts)
    coor_b[:,0,:] = fa1p00+fa1p10*coor_3d[:,2,:]+fa1p01*coor_3d[:,0,:]+fa1p20*coor_3d[:,2,:]**2+fa1p11*coor_3d[:,2,:]*coor_3d[:,0,:]+fa1p30*coor_3d[:,2,:]**3+fa1p21*coor_3d[:,2,:]**2*coor_3d[:,0,:]
    coor_b[:,1,:] = fa2p00+fa2p10*coor_3d[:,2,:]+fa2p01*coor_3d[:,1,:]+fa2p20*coor_3d[:,2,:]**2+fa2p11*coor_3d[:,2,:]*coor_3d[:,1,:]+fa2p30*coor_3d[:,2,:]**3+fa2p21*coor_3d[:,2,:]**2*coor_3d[:,1,:]
    coor_s1[:,0,:] = fb1p00+fb1p10*coor_3d[:,0,:]+fb1p01*coor_3d[:,1,:]+fb1p20*coor_3d[:,0,:]**2+fb1p11*coor_3d[:,0,:]*coor_3d[:,1,:]+fb1p30*coor_3d[:,0,:]**3+fb1p21*coor_3d[:,0,:]**2*coor_3d[:,1,:]
    coor_s1[:,1,:] = fb2p00+fb2p10*coor_3d[:,0,:]+fb2p01*coor_3d[:,2,:]+fb2p20*coor_3d[:,0,:]**2+fb2p11*coor_3d[:,0,:]*coor_3d[:,2,:]+fb2p30*coor_3d[:,0,:]**3+fb2p21*coor_3d[:,0,:]**2*coor_3d[:,2,:]
    coor_s2[:,0,:] = fc1p00+fc1p10*coor_3d[:,1,:]+fc1p01*coor_3d[:,0,:]+fc1p20*coor_3d[:,1,:]**2+fc1p11*coor_3d[:,1,:]*coor_3d[:,0,:]+fc1p30*coor_3d[:,1,:]**3+fc1p21*coor_3d[:,1,:]**2*coor_3d[:,0,:]
    coor_s2[:,1,:] = fc2p00+fc2p10*coor_3d[:,1,:]+fc2p01*coor_3d[:,2,:]+fc2p20*coor_3d[:,1,:]**2+fc2p11*coor_3d[:,1,:]*coor_3d[:,2,:]+fc2p30*coor_3d[:,1,:]**3+fc2p21*coor_3d[:,1,:]**2*coor_3d[:,2,:]
    return coor_b, coor_s1, coor_s2

coor_b, coor_s1, coor_s2 = calc_proj_w_refra(coor_3d,proj_params)
print(torch.sum(coor_b))
print(torch.sum(coor_s1))
print(torch.sum(coor_s2))
