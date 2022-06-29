% Last modified by Aniket on 29th of April (Line 30)

function [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,annotated_b,annotated_s1,annotated_s2,eye_b,eye_s1,eye_s2,coor_3d] =...
    return_graymodels_fish(x,lut_b_tail,lut_s_tail,proj_params,fishlen,imageSizeX,imageSizeY)
% initial guess of the position
% seglen is the length of each segment
seglen = fishlen*0.09;
% alpha: azimuthal angle of the rotated plane
% gamma: direction cosine of the plane of the fish with z-axis
% theta: angles between segments along the plane with direction cosines
% alpha, beta and gamma
hp = [x(1);x(2);x(3)];
dtheta = x(4:12);
theta = cumsum(dtheta);
dphi = x(13:21);
phi = cumsum(dphi);
vec = seglen*[cos(theta).*cos(phi); sin(theta).*cos(phi); -sin(phi)];

% vec_ref_1 is parallel to the camera sensor of b and s2
% vec_ref_2 is parallel to s1
vec_ref_1 = [seglen;0;0];
vec_ref_2 = [0;seglen;0];
pt_ref = [hp + vec_ref_1, hp + vec_ref_2];

pt = cumsum([[0;0;0],vec],2);
pt = [pt + repmat(hp,1,10), pt_ref];

% use cen_3d as the 4th point on fish
hinge = pt(:,3);
vec_13 = pt(:,1) - hinge;
vec_13 = repmat(vec_13,1,12);

pt = pt + vec_13;

[coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu(pt,proj_params);

% keep the corresponding vec_ref for each 
coor_b(:,end-1:end) = [];
coor_s1(:,end-1) = [];
coor_s2(:,end) = [];

% Re-defining cropped coordinates for training images of dimensions
% imageSizeY x imageSizeX
crop_b(1) = round(coor_b(2,3)) - (imageSizeY - 1)/2;
crop_b(2) = crop_b(1) + imageSizeY - 1;
crop_b(3) = round(coor_b(1,3)) - (imageSizeX - 1)/2;
crop_b(4) = crop_b(3) + imageSizeX - 1;

crop_s1(1) = round(coor_s1(2,3)) - (imageSizeY - 1)/2;
crop_s1(2) = crop_s1(1) + imageSizeY - 1;
crop_s1(3) = round(coor_s1(1,3)) - (imageSizeX - 1)/2;
crop_s1(4) = crop_s1(3) + imageSizeX - 1;

crop_s2(1) = round(coor_s2(2,3)) - (imageSizeY - 1)/2;
crop_s2(2) = crop_s2(1) + imageSizeY - 1;
crop_s2(3) = round(coor_s2(1,3)) - (imageSizeX - 1)/2;
crop_s2(4) = crop_s2(3) + imageSizeX - 1;

annotated_b(1,:) = coor_b(1,:) - crop_b(3) + 1;
annotated_b(2,:) = coor_b(2,:) - crop_b(1) + 1;
annotated_s1(1,:) = coor_s1(1,:) - crop_s1(3) + 1;
annotated_s1(2,:) = coor_s1(2,:) - crop_s1(1) + 1;
annotated_s2(1,:) = coor_s2(1,:) - crop_s2(3) + 1;
annotated_s2(2,:) = coor_s2(2,:) - crop_s2(1) + 1;

annotated_b = annotated_b(:,1:10);
annotated_s1 = annotated_s1(:,1:10);
annotated_s2 = annotated_s2(:,1:10);

[projection_b,projection_s1,projection_s2,eye_b,eye_s1,eye_s2,eye_coor_3d] = ...
    return_head_real_model_new(x,fishlen,proj_params,crop_b,crop_s1,crop_s2);

eye_b(1,:) = eye_b(1,:) - crop_b(3) + 1;
eye_b(2,:) = eye_b(2,:) - crop_b(1) + 1;
eye_s1(1,:) = eye_s1(1,:) - crop_s1(3) + 1;
eye_s1(2,:) = eye_s1(2,:) - crop_s1(1) + 1;
eye_s2(1,:) = eye_s2(1,:) - crop_s2(3) + 1;
eye_s2(2,:) = eye_s2(2,:) - crop_s2(1) + 1;

gray_b = view_b_lut_new_real_cpu(crop_b,coor_b,lut_b_tail,projection_b,imageSizeX,imageSizeY);
gray_s1 = view_s_lut_new_real_cpu(crop_s1,coor_s1,lut_s_tail,projection_s1,imageSizeX,imageSizeY);
gray_s2 = view_s_lut_new_real_cpu(crop_s2,coor_s2,lut_s_tail,projection_s2,imageSizeX,imageSizeY);

imblurred = imgaussfilt(gray_b, 1);
thresh_bw = graythresh(imblurred(imblurred>20))*0.42;
bw = imbinarize(imblurred,thresh_bw);
bw = imerode(bw,ones(2));
bw = bwareaopen(bw, 100);
cen = regionprops(bw, 'Centroid');

% Subtract 1 for accordance with python's format
eye_b = eye_b - 1; eye_s1 = eye_s1 - 1; eye_s2 = eye_s2 - 1;
annotated_b = annotated_b - 1; annotated_s1 = annotated_s1 - 1; annotated_s2 = annotated_s2 - 1;
crop_b = crop_b - 1; crop_s1 = crop_s1 - 1; crop_s2 = crop_s2 - 1;
coor_3d = pt(:,1:10);
coor_3d = [coor_3d,eye_coor_3d];
end
