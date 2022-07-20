function [belly_model] = bellymodel(x, y, z, seglen, brightness, size_lut)

%belly_w = seglen * (0.3527); %0.5030;5527
belly_w = seglen * (0.489 + (rand - 0.5)*0.03);
belly_l = seglen * (1.2500 + (rand - 0.5)*0.07); %1.5970; 
%belly_h = seglen * (0.6531); %0.6294; % 0.55
%belly_h = seglen * (0.7431); 
belly_h = seglen * (0.7231 + (rand - 0.5)*0.03);
c_belly = 1.0541 + (rand - 0.5)*0.03;
% R = rotz(heading)*roty(inclination)*rotx(roll);
pt_original(:,2) = [size_lut/2; size_lut/2; size_lut/2];
pt_original(:,1) = pt_original(:,2) - [seglen; 0; 0];
pt_original(:,3) = pt_original(:,2) + [seglen; 0; 0];
belly_c = [c_belly*pt_original(1,2) + (1-c_belly)*pt_original(1,3); c_belly*pt_original(2,2) + ...
    (1-c_belly)*pt_original(2,3); pt_original(3,2) - seglen/(6+(rand - 0.5)*0.05)]; %3.5602]; % Changed from 6   7.0257
% belly_c = belly_c - pt_original(:,2);
% belly_c = R*belly_c  + pt_original(:,2);

XX = x - belly_c(1);
YY = y - belly_c(2);
ZZ = z - belly_c(3);
% rot_mat = rotx(-roll)*roty(-inclination)*rotz(-heading);
% XX = rot_mat(1,1)*XX + rot_mat(1,2)*YY + rot_mat(1,3)*ZZ;
% YY = rot_mat(2,1)*XX + rot_mat(2,2)*YY + rot_mat(2,3)*ZZ;
% ZZ = rot_mat(3,1)*XX + rot_mat(3,2)*YY + rot_mat(3,3)*ZZ;

belly_model = exp(-2*(XX.*XX/(2*belly_l^2) + YY.*YY/(2*belly_w^2) + ...
    ZZ.*ZZ/(2*belly_h^2) - 1));
belly_model = belly_model*brightness;
% belly_model(find(belly_model <= 6)) = 0;
% log_indices = belly_model > 12;
% belly_model = belly_model.*log_indices;
