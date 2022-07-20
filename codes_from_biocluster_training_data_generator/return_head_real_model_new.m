%Written by Aniket Ravan
% 5th of May 2019
% Last edit on 4th of September 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [graymodel_b,graymodel_s1,graymodel_s2,eye_b,eye_s1,eye_s2,eye_3d_coor] = return_head_real_model_new(x,fishlen,proj_params,cb,cs1,cs2)


% Calculate the 3D points pt from model parameters
seglen = fishlen * 0.09;
size_lut_3d = 2; % Represents the length of the box in which the 3D fish is constructed
inclination = x(13);
heading = x(4);
hp = [x(1);x(2);x(3)];
dtheta = x(4:12);
theta = cumsum(dtheta);
dphi = x(13:21);
phi = cumsum(dphi);
roll = x(22);

vec_unit = seglen*[cos(theta).*cos(phi); sin(theta).*cos(phi); -sin(phi)];

% vec_ref_1 is parallel to the camera sensor of b and s2
% vec_ref_2 is parallel to s1
vec_ref_1 = [seglen;0;0];
vec_ref_2 = [0;seglen;0];
pt_ref = [hp + vec_ref_1, hp + vec_ref_2];
pt = cumsum([[0;0;0],vec_unit],2);
pt = [pt + repmat(hp,1,10), pt_ref];

% Construct the larva
% Locate center of the head to use as origin for rotation of the larva.
% This is consistent with the way in which the parameters of the model are
% computed during optimization
resolution = 75;
x_c = (linspace(0, 1, resolution) * size_lut_3d);
y_c = (linspace(0, 1, resolution) * size_lut_3d);
z_c = (linspace(0, 1, resolution) * size_lut_3d);
[x_c,y_c,z_c] = meshgrid(x_c,y_c,z_c);
x_c = x_c(:)'; y_c = y_c(:)'; z_c = z_c(:)';
pt_original(:,2) = [size_lut_3d/2; size_lut_3d/2; size_lut_3d/2];
pt_original(:,1) = pt_original(:,2) - [seglen; 0; 0];
pt_original(:,3) = pt_original(:,2) + [seglen; 0; 0];
hinge = pt_original(:,2); % COM of the fish
% Calculate the 3D fish
% eye_br = 120.8125; % 150.8125;
% head_br = 15.953318957123471;
% belly_br = 17.05897936938326;
% eye_br = eye_br*4.1; head_br = head_br*3.1; belly_br = belly_br*3.6;
% %eye_br = eye_br*0.033; head_br = head_br*0.021; belly_br = belly_br*0.028;
%%% Translate the model to overlap with the cropped image
vec_13 = pt(:,1) - pt(:,3);
vec_13 = repmat(vec_13,1,12);
pt = pt + vec_13;
ref_vec = pt(:,2) - hinge;
eye_br = 13; head_br = 13; belly_br = 13;
% Render and project the 3D fish

random_vector_eye = rand([1 5]);
[eye1_model,eye1_c] = eye1model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye);
[eye2_model,eye2_c] = eye2model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye);
belly_model = bellymodel(x_c, y_c, z_c, seglen, belly_br, size_lut_3d);
head_model = headmodel(x_c, y_c, z_c, seglen, head_br, size_lut_3d);
% model = max(max(max(eye1_model,eye2_model),head_model),belly_model);
% [model_X, model_Y, model_Z, indices] = reorient_model(model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
% [graymodel_b, graymodel_s1, graymodel_s2] = project_camera_copy(model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

% Eye1
[model_X, model_Y, model_Z, indices] = reorient_model(eye1_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
[eye1_b, eye1_s1, eye1_s2] = project_camera_copy(eye1_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

% Eye2
[model_X, model_Y, model_Z, indices] = reorient_model(eye2_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
[eye2_b, eye2_s1, eye2_s2] = project_camera_copy(eye2_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

% Head
[model_X, model_Y, model_Z, indices] = reorient_model(head_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
[head_b, head_s1, head_s2] = project_camera_copy(head_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

% Bellymodel
[model_X, model_Y, model_Z, indices] = reorient_model(belly_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
[belly_b, belly_s1, belly_s2] = project_camera_copy(belly_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

eye_br = 112 + (rand - 0.5)*5; head_br = 72 + (rand - 0.5)*5; belly_br = 83 + (rand - 0.5)*5;
eye_scaling = max(eye_br/double(max(max(eye1_b))),eye_br/double(max(max(eye2_b))));
eye1_b = eye1_b*(eye_scaling);
eye2_b = eye2_b*(eye_scaling);
head_b = head_b*(head_br/double(max(max(head_b))));
belly_b = belly_b*(belly_br/double(max(max(belly_b))));

eye_scaling = max(eye_br/double(max(max(eye1_s1))),eye_br/double(max(max(eye2_s1))));
eye1_s1 = eye1_s1*(eye_scaling);
eye2_s1 = eye2_s1*(eye_br/double(max(max(eye2_s1))));
head_s1 = head_s1*(head_br/double(max(max(head_s1))));
belly_s1 = belly_s1*(belly_br/double(max(max(belly_s1))));

eye_scaling = max(eye_br/double(max(max(eye1_s2))),eye_br/double(max(max(eye2_s2))));
eye1_s2 = eye1_s2*(eye_scaling);
eye2_s2 = eye2_s2*(eye_scaling);
head_s2 = head_s2*(head_br/double(max(max(head_s2))));
belly_s2 = belly_s2*(belly_br/double(max(max(belly_s2))));

graymodel_b = max(max(max(eye1_b,eye2_b),head_b),belly_b);
graymodel_s1 = max(max(max(eye1_s1,eye2_s1),head_s1),belly_s1);
graymodel_s2 = max(max(max(eye1_s2,eye2_s2),head_s2),belly_s2);

[eyeCenters_X, eyeCenters_Y, eyeCenters_Z] =...
    reorient_model([],[eye1_c(1),eye2_c(1)],[eye1_c(2),eye2_c(2)],[eye1_c(3),eye2_c(3)],heading,inclination,roll,ref_vec,hinge);
[eye_b,eye_s1,eye_s2] = calc_proj_w_refra_cpu([eyeCenters_X;eyeCenters_Y;eyeCenters_Z], proj_params);
eye_3d_coor = [eyeCenters_X; eyeCenters_Y; eyeCenters_Z];
graymodel_b = (medfilt2(graymodel_b)); 
graymodel_s1 = (medfilt2(graymodel_s1));
graymodel_s2 = (medfilt2(graymodel_s2));
