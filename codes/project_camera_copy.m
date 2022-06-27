%Written by Aniket Ravan
% 5th of May 2019
% Last edit on 4th of September 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [projection_b,projection_s1,projection_s2] = project_camera_copy(model, X, Y, Z, proj_params, indices, cb, cs1, cs2)


[coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu([X; Y; Z], proj_params);

coor_b(1,:) = coor_b(1,:) - cb(3) + 1; coor_b(2,:) = coor_b(2,:) - cb(1) + 1;
coor_s1(1,:) = coor_s1(1,:) - cs1(3) + 1; coor_s1(2,:) = coor_s1(2,:) - cs1(1) + 1;
coor_s2(1,:) = coor_s2(1,:) - cs2(3) + 1; coor_s2(2,:) = coor_s2(2,:) - cs2(1) + 1;
%projection_b = zeros(size_lut_2d_b,size_lut_2d_b);
%projection_s1 = zeros(size_lut_2d_s,size_lut_2d_s);
%projection_s2 = zeros(size_lut_2d_s,size_lut_2d_s);

%projection_b = zeros(488,648);
%projection_s1 = zeros(488,648);
%projection_s2 = zeros(488,648);

projection_b = zeros(cb(2) - cb(1) + 1, cb(4) - cb(3) + 1);
projection_s1 = zeros(cs1(2) - cs1(1) + 1, cs1(4) - cs1(3) + 1);
projection_s2 = zeros(cs2(2) - cs2(1) + 1, cs2(4) - cs2(3) + 1);

sz_b = size(projection_b); sz_s1 = size(projection_s1); sz_s2 = size(projection_s2);
count_mat_b = zeros(size(projection_b)) + 0.0001;
count_mat_s1 = zeros(size(projection_s1)) + 0.0001;
count_mat_s2 = zeros(size(projection_s2)) + 0.0001;

        %coor_b(1,:) = coor_b(1,:) - pt2_b(1) + ceil(size_lut_2d_b/2);
        %coor_b(2,:) = coor_b(2,:) - pt2_b(2) + ceil(size_lut_2d_b/2);
        for i = 1:length(indices)
            if (floor(coor_b(2,i)) > sz_b(1) ||  floor(coor_b(1,i)) > sz_b(2) || floor(coor_b(2,i)) < 1 || floor(coor_b(1,i)) < 1)
                continue
            end
            projection_b(floor(coor_b(2,i)), floor(coor_b(1,i))) = ...
                projection_b(floor(coor_b(2,i)), floor(coor_b(1,i))) + model(indices(i));
            count_mat_b(floor(coor_b(2,i)), floor(coor_b(1,i))) = ...
                count_mat_b(floor(coor_b(2,i)), floor(coor_b(1,i))) + 1;
            
        end
        projection_b = projection_b./count_mat_b;
        
  
	%coor_s1(1,:) = coor_s1(1,:) - pt2_s1(1) + ceil(size_lut_2d_s/2);
        %coor_s1(2,:) = coor_s1(2,:) - pt2_s1(2) + ceil(size_lut_2d_s/2);
        for i = 1:length(indices)
            if (floor(coor_s1(2,i)) > sz_s1(1) ||  floor(coor_s1(1,i)) > sz_s1(2) || floor(coor_s1(2,i)) < 1 || floor(coor_s1(1,i)) < 1)
                continue
            end
            projection_s1(floor(coor_s1(2,i)), floor(coor_s1(1,i))) = ...
                projection_s1(floor(coor_s1(2,i)), floor(coor_s1(1,i))) + model(indices(i));
            count_mat_s1(floor(coor_s1(2,i)), floor(coor_s1(1,i))) = ...
                count_mat_s1(floor(coor_s1(2,i)), floor(coor_s1(1,i))) + 1;
        end
        projection_s1 = projection_s1./count_mat_s1;

       
        %coor_s2(1,:) = coor_s2(1,:) - pt2_s2(1) + ceil(size_lut_2d_s/2);;
        %coor_s2(2,:) = coor_s2(2,:) - pt2_s2(2) + ceil(size_lut_2d_s/2);;
        for i = 1:length(indices)
            if (floor(coor_s2(2,i)) > sz_s2(1) ||  floor(coor_s2(1,i)) > sz_s2(2) || floor(coor_s2(2,i)) < 1 || floor(coor_s2(1,i)) < 1)
                continue
            end
            projection_s2(floor(coor_s2(2,i)), floor(coor_s2(1,i))) = ...
                projection_s2(floor(coor_s2(2,i)), floor(coor_s2(1,i))) + model(indices(i));
            count_mat_s2(floor(coor_s2(2,i)), floor(coor_s2(1,i))) = ...
                count_mat_s2(floor(coor_s2(2,i)), floor(coor_s2(1,i))) + 1;
        end
        projection_s2 = projection_s2./count_mat_s2;




