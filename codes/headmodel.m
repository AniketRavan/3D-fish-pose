function [head_model] = headmodel(x, y, z, seglen, brightness, size_lut)

head_w = seglen * (0.7462); %0.8197;  % 0.6
head_l = seglen * (0.7475); %0.8196; % 1/sqrt(2)
head_h = seglen * (0.6426); %0.7622;  % 0.7/sqrt(2)
c_head = 1.3371; %1.1296;
pt_original(:,2) = [size_lut/2; size_lut/2 ; size_lut/2];
pt_original(:,1) = pt_original(:,2) - [seglen; 0; 0];
pt_original(:,3) = pt_original(:,2) + [seglen; 0; 0];
% R = rotz(heading)*roty(inclination)*rotx(roll);
head_c = [c_head*pt_original(1,1) + (1-c_head)*pt_original(1,2); c_head*pt_original(2,1) + ...
    (1-c_head)*pt_original(2,2); pt_original(3,2) - seglen/9.4590];%3.4609];
% head_c = head_c - pt_original(:,2);
% head_c = R*head_c + pt_original(:,2);

XX = x - head_c(1);
YY = y - head_c(2);
ZZ = z - head_c(3);
% rot_mat = rotx(-roll)*roty(-inclination)*rotz(-heading);
% XX = rot_mat(1,1)*XX + rot_mat(1,2)*YY + rot_mat(1,3)*ZZ;
% YY = rot_mat(2,1)*XX + rot_mat(2,2)*YY + rot_mat(2,3)*ZZ;
% ZZ = rot_mat(3,1)*XX + rot_mat(3,2)*YY + rot_mat(3,3)*ZZ;

head_model = exp(-2*(XX.*XX/(2*head_l^2) + YY.*YY/(2*head_w^2) + ...
    ZZ.*ZZ/(2*head_h^2) - 1));
head_model = head_model*brightness;
% head_model(find(head_model <= 6)) = 0;
% log_indices = head_model > 12;
% head_model = head_model.*log_indices;
