function [eye1_model,eye1_c] = eye1model(x, y, z, seglen, brightness, size_lut)
d_eye = seglen * 0.9056;% 1.4332; %0.83
c_eyes = 1.4130; %1.3015;
eye1_w = seglen * 0.2097; %0.1671; % 0.35
eye1_l = seglen * 0.3006; %0.2507; % 0.45
eye1_h = seglen * 0.2296; %0.2661; % 0.35
% R = rotz(heading)*roty(inclination)*rotx(roll);
pt_original(:,2) = [size_lut/2; size_lut/2; size_lut/2];
pt_original(:,1) = pt_original(:,2) - [seglen; 0; 0];
pt_original(:,3) = pt_original(:,2) + [seglen; 0; 0];
eye1_c = [c_eyes*pt_original(1,1) + (1-c_eyes)*pt_original(1,2); c_eyes*pt_original(2,1) + ...
    (1-c_eyes)*pt_original(2,2) + d_eye/2; pt_original(3,2) - seglen/7.3049];%2.9028
% eye1_c = eye1_c - pt_original(:,2);
% eye1_c = R*eye1_c + pt_original(:,2);

XX = x - eye1_c(1);
YY = y - eye1_c(2);
ZZ = z - eye1_c(3);
% rot_mat = rotx(-roll)*roty(-inclination)*rotz(-heading);
% XX = rot_mat(1,1)*XX + rot_mat(1,2)*YY + rot_mat(1,3)*ZZ;
% YY = rot_mat(2,1)*XX + rot_mat(2,2)*YY + rot_mat(2,3)*ZZ;
% ZZ = rot_mat(3,1)*XX + rot_mat(3,2)*YY + rot_mat(3,3)*ZZ;

eye1_model = exp(-1.2*(XX.*XX/(2*eye1_l^2) + YY.*YY/(2*eye1_w^2) + ...
    ZZ.*ZZ/(2*eye1_h^2) - 1));
eye1_model = eye1_model*brightness;
% eye1_model(find(eye1_model <= 6)) = 0;
% log_indices = eye1_model > 12;
% eye1_model = eye1_model.*log_indices;
