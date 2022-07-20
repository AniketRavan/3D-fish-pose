% Author: Aniket Ravan
% Generates training dataset for side views 
% Last modified: 12th of May, 2022

load('../proj_params_101019_corrected_new')
load('../lut_b_tail')
load('../lut_s_tail')
path{1} = '../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 400000;
x_complete = [];
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
mkdir('../training_data_3D_pose_new/annotations_220623_pose')
mkdir('../training_data_3D_pose_new/annotations_220623_crop_coor')
mkdir('../training_data_3D_pose_new/annotations_220623_eye_coor')
mkdir('../training_data_3D_pose_new/images')
mkdir('../training_data_3D_pose_new/annotations_220623_coor_3d')
for path_idx = 1
    coor_mf_mats = dir(path{path_idx});
    for z = length(coor_mf_mats) - 2:-1:round(3*((length(coor_mf_mats)) - 2)/4)
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        [num2str(z),' of ',num2str(length(coor_mf_mats))]
        for i = 1:length(coor_mat_mf.x_all)
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            
            for n = 1:nframes
                if (n/nframes > 0.15 && n/nframes < 0.75 && path_idx == 1)
                    n_iterations = 17;
                elseif (n/nframes > 0.15 && n/nframes < 0.75 && path_idx > 1)
                    n_iterations = 17;
                else
                    continue
                end
                for iter = 1:n_iterations
                    % model fish image
                    x = x_all_mf(n,:);
                    x(4) = rand*2*pi;
                    x(13) = normrnd(0,0.3)*pi/2;                                       
                    [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                    eye_coor = [eye_b,eye_s1,eye_s2];
		    gray_b = imnoise(gray_b, 'gaussian', 0,  0.0007*(rand+1));
                    gray_s1 = imnoise(gray_s1, 'gaussian', 0,  0.0007*(rand+1));
                    gray_s2 = imnoise(gray_s2, 'gaussian', 0,  0.0007*(rand+1));
%                     im_tail = imnoise(im_tail, 'gaussian', 0,  0.0007*(rand+1));
                    gray_b = gray_b*(255/double(max(gray_b(:))));
                    gray_s1 = gray_s1*(255/double(max(gray_s1(:))));
                    gray_s2 = gray_s2*(255/double(max(gray_s2(:))));
                    im = cat(1,gray_b,gray_s1,gray_s2);
%                     im_tail = im_tail*(255/double(max(gray_b(:))));
                    imwrite(im,['../training_data_3D_pose_new/images/im_v_2',num2str(idx),'.png'])                  
%                     imwrite(im_tail,['training_side_tail\im_',num2str(idx),'_',num2str(n),'.png'])
                    crop_coor = [crop_b, crop_s1, crop_s2];
                    pose = [c_b, c_s1, c_s2];
		    if (isempty(eye_coor) || isempty(coor_3d) || isempty(pose))
		        display(x)
			display(coor_mf_matname)
		    end
		    save(['../training_data_3D_pose_new/annotations_220623_coor_3d/coor_3d_',num2str(idx),'.mat'],  'coor_3d');
                    save(['../training_data_3D_pose_new/annotations_220623_pose/pose_ann_',num2str(idx),'.mat'],  'pose');
                    save(['../training_data_3D_pose_new/annotations_220623_crop_coor/crop_coor_ann_',num2str(idx),'.mat'],  'crop_coor');
		    save(['../training_data_3D_pose_new/annotations_220623_eye_coor/eye_coor_ann_',num2str(idx),'.mat'],  'eye_coor');
                    idx = idx + 1;
		    if (idx == 500000)
			display('Finished 500000')
		        return
		    end
                end
            end
        end
    end
end
% csvwrite('annotations_3D.csv', crop_coor_mat)
