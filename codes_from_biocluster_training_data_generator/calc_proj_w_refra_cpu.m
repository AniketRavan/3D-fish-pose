% This function does triangulation and refraction calibration
% the input is the 3D coordinates of the model
% the output is the 2D coordinates in three views after refraction
% camera a corresponds to view s1
% camera b corresponds to b
% camera c corresponds to s2

% The center of the tank is (35.2,27.5,35.7);
% The sensor of camera a is parallel to x-y plane
% camera b parallel to x-z
% camera c parallel to y-z

function [coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu(coor_3d,proj_params)

fa1p00 = proj_params(1,1);
fa1p10 = proj_params(1,2);
fa1p01 = proj_params(1,3);
fa1p20 = proj_params(1,4);
fa1p11 = proj_params(1,5);
fa1p30 = proj_params(1,6);
fa1p21 = proj_params(1,7);
fa2p00 = proj_params(2,1);
fa2p10 = proj_params(2,2);
fa2p01 = proj_params(2,3);
fa2p20 = proj_params(2,4);
fa2p11 = proj_params(2,5);
fa2p30 = proj_params(2,6);
fa2p21 = proj_params(2,7);
fb1p00 = proj_params(3,1);
fb1p10 = proj_params(3,2);
fb1p01 = proj_params(3,3);
fb1p20 = proj_params(3,4);
fb1p11 = proj_params(3,5);
fb1p30 = proj_params(3,6);
fb1p21 = proj_params(3,7);
fb2p00 = proj_params(4,1);
fb2p10 = proj_params(4,2);
fb2p01 = proj_params(4,3);
fb2p20 = proj_params(4,4);
fb2p11 = proj_params(4,5);
fb2p30 = proj_params(4,6);
fb2p21 = proj_params(4,7);
fc1p00 = proj_params(5,1);
fc1p10 = proj_params(5,2);
fc1p01 = proj_params(5,3);
fc1p20 = proj_params(5,4);
fc1p11 = proj_params(5,5);
fc1p30 = proj_params(5,6);
fc1p21 = proj_params(5,7);
fc2p00 = proj_params(6,1);
fc2p10 = proj_params(6,2);
fc2p01 = proj_params(6,3);
fc2p20 = proj_params(6,4);
fc2p11 = proj_params(6,5);
fc2p30 = proj_params(6,6);
fc2p21 = proj_params(6,7);

npts = size(coor_3d,2);
coor_b = zeros(2,npts);
coor_s1 = zeros(2,npts);
coor_s2 = zeros(2,npts);

%for n = 1:npts
%    x = coor_3d(:,n);
coor_b(1,:) = fa1p00+fa1p10*coor_3d(3,:)+fa1p01*coor_3d(1,:)+fa1p20*coor_3d(3,:).^2+fa1p11*coor_3d(3,:).*coor_3d(1,:)+fa1p30*coor_3d(3,:).^3+fa1p21*coor_3d(3,:).^2.*coor_3d(1,:);
coor_b(2,:) = fa2p00+fa2p10*coor_3d(3,:)+fa2p01*coor_3d(2,:)+fa2p20*coor_3d(3,:).^2+fa2p11*coor_3d(3,:).*coor_3d(2,:)+fa2p30*coor_3d(3,:).^3+fa2p21*coor_3d(3,:).^2.*coor_3d(2,:);
coor_s1(1,:) = fb1p00+fb1p10*coor_3d(1,:)+fb1p01*coor_3d(2,:)+fb1p20*coor_3d(1,:).^2+fb1p11*coor_3d(1,:).*coor_3d(2,:)+fb1p30*coor_3d(1,:).^3+fb1p21*coor_3d(1,:).^2.*coor_3d(2,:);
coor_s1(2,:) = fb2p00+fb2p10*coor_3d(1,:)+fb2p01*coor_3d(3,:)+fb2p20*coor_3d(1,:).^2+fb2p11*coor_3d(1,:).*coor_3d(3,:)+fb2p30*coor_3d(1,:).^3+fb2p21*coor_3d(1,:).^2.*coor_3d(3,:);
coor_s2(1,:) = fc1p00+fc1p10*coor_3d(2,:)+fc1p01*coor_3d(1,:)+fc1p20*coor_3d(2,:).^2+fc1p11*coor_3d(2,:).*coor_3d(1,:)+fc1p30*coor_3d(2,:).^3+fc1p21*coor_3d(2,:).^2.*coor_3d(1,:);
coor_s2(2,:) = fc2p00+fc2p10*coor_3d(2,:)+fc2p01*coor_3d(3,:)+fc2p20*coor_3d(2,:).^2+fc2p11*coor_3d(2,:).*coor_3d(3,:)+fc2p30*coor_3d(2,:).^3+fc2p21*coor_3d(2,:).^2.*coor_3d(3,:);
end


% X0 = coor_3d';

% % new coordinate system
% X = zeros(npts,3);
% X(:,1) = -X0(:,1)+9.6;
% X(:,2) = -X0(:,2)+49.5;
% X(:,3) = X0(:,3)+145.1;
% 
% 
% depth_s1 = X(:,3);
% depth_b = X(:,2);
% depth_s2 = X(:,1);


