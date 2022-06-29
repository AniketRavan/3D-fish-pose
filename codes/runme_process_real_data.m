% Author: Aniket Ravan
% Generates real dataset for 3-D pose estimation
% Last modified: 27rd of June, 2022
 
load('proj_params_101019_corrected_new')
load('lut_b_tail')
load('lut_s_tail')
path{1} = '../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 0;
x_complete = [];
swimtype = 'er';
date = '220629';
imageSizeX = 141; imageSizeY = 141;
mkdir(['../validation_data_3D_pose_shifted/annotations_',date,'_crop_coor'])
mkdir(['../validation_data_3D_pose_shifted/images'])
mkdir(['../validation_data_3D_pose_shifted/annotations_',date,'_coor_3d'])
display('Created directories')
for path_idx = 1
    coor_mf_mats = dir(path{path_idx});
    for z = 1:(length(coor_mf_mats)) - 2
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        [num2str(z),' of ',num2str(length(coor_mf_mats))]
        for i = 1:length(coor_mat_mf.fish_in_vid.b)
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            
            for n = 1:nframes
                x = x_all_mf(n,:);
                im_b = coor_mat_mf.fish_in_vid.b{i}{1}{n};
                im_s1 = coor_mat_mf.fish_in_vid.s1{i}{1}{n};
                im_s2 = coor_mat_mf.fish_in_vid.s2{i}{1}{n};
                buffer_x = round((imageSizeX - size(im_b,2))/2);
                buffer_y = round((imageSizeY - size(im_b,1))/2);
                im_b_full = zeros(imageSizeY,imageSizeX,'uint8');
                im_b_full = imnoise(im_b_full, 'gaussian', 0,  0.00007*(rand+1));
                im_s1_full = im_b_full;
                im_s1_full = imnoise(im_s1_full, 'gaussian', 0,  0.00007*(rand+1));
                im_s2_full = im_b_full;
                im_s2_full = imnoise(im_s2_full, 'gaussian', 0,  0.00007*(rand+1));
                im_b_full(buffer_y+1:buffer_y+size(im_b,1), buffer_x+1:buffer_x+size(im_b,2)) = ...
                    im_b;
                im_b_full = im_b_full*(255/double(max(im_b_full(:))));
                crop_b = coor_mat_mf.fish_in_vid.b{i}{2}{n};
                crop_s1 = coor_mat_mf.fish_in_vid.s1{i}{2}{n};
                crop_s2 = coor_mat_mf.fish_in_vid.s2{i}{2}{n};
                [~,~,~,~,~,~,~,~,~,~,~,~,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
		crop_b(1) = crop_b(1) - buffer_y;
                crop_b(2) = crop_b(1) + imageSizeY - 1;
                crop_b(3) = crop_b(3) - buffer_x;
                crop_b(4) = crop_b(4) + imageSizeX - 1;
                
                buffer_x = round((imageSizeX - size(im_s1,2))/2);
                buffer_y = round((imageSizeY - size(im_s1,1))/2);
                im_s1_full(buffer_y+1:buffer_y+size(im_s1,1), buffer_x+1:buffer_x+size(im_s1,2)) = ...
                    im_s1;
                im_s1_full = im_s1_full*(255/double(max(im_s1_full(:))));
                crop_s1(1) = crop_s1(1) - buffer_y;
                crop_s1(2) = crop_s1(1) + imageSizeY - 1;
                crop_s1(3) = crop_s1(3) - buffer_x;
                crop_s1(4) = crop_s1(4) + imageSizeX - 1;
                
                buffer_x = round((imageSizeX - size(im_s2,2))/2);
                buffer_y = round((imageSizeY - size(im_s2,1))/2);
                im_s2_full(buffer_y+1:buffer_y+size(im_s2,1), buffer_x+1:buffer_x+size(im_s2,2)) = ...
                    im_s2;
                im_s2_full = im_s2_full*(255/double(max(im_s2_full(:))));
                crop_s2(1) = crop_s2(1) - buffer_y;
                crop_s2(2) = crop_s2(1) + imageSizeY - 1;
                crop_s2(3) = crop_s2(3) - buffer_x;
                crop_s2(4) = crop_s2(4) + imageSizeX - 1;
                whos coor_3d
                im = cat(1,im_b_full,im_s1_full,im_s2_full);
                crop_coor = [crop_b, crop_s1, crop_s2];
                imwrite(im,['../validation_data_3D_pose_shifted/images/im_',num2str(idx),'.png'])
                save(['../validation_data_3D_pose_shifted/annotations_',date,'_crop_coor/crop_coor_ann_',num2str(idx),'.mat'],'crop_coor');  
                save(['../validation_data_3D_pose_shifted/annotations_',date,'_coor_3d/coor_3d_ann_',num2str(idx),'.mat'],'coor_3d');
		idx = idx + 1;               
            end
        end
    end
end


