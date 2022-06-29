% Author: Aniket Ravan
% Generates training dataset for side views 
% Last modified: 12th of May, 2022

load('../proj_params_101019_corrected_new')
load('../lut_b_tail')
load('../lut_s_tail')
path{1} = '../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 0;
x_complete = [];
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
date = '220626';
mkdir(['../training_data_3D_pose_new/annotations_',date,'_pose'])
mkdir(['../training_data_3D_pose_new/annotations_',date,'_crop_coor'])
mkdir(['../training_data_3D_pose_new/annotations_',date,'_eye_coor'])
mkdir(['../training_data_3D_pose_new/images'])
mkdir(['../training_data_3D_pose_new/annotations_',date,'_coor_3d'])
for path_idx = 1
    coor_mf_mats = dir(path{path_idx});
    for z = randi([1 (length(coor_mf_mats))-2])
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        [num2str(z),' of ',num2str(length(coor_mf_mats)- 2)]
        for i = randi([1 length(coor_mat_mf.x_all)])
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            
            for n = randi([1 nframes])
                    x = x_all_mf(n,:);
                    [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                    eye_coor = [eye_b,eye_s1,eye_s2];
		    gray_b = imgaussfilt(gray_b); gray_s1 = imgaussfilt(gray_s1); gray_s2 = imgaussfilt(gray_s2);
		    gray_b = imnoise(gray_b, 'gaussian', (rand*3+7)/255,  (rand*10+40)/255^2); %mean in range(7,10), var in range(40,50)
                    gray_s1 = imnoise(gray_s1, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    gray_b = gray_b*(255/double(max(gray_b(:))));
                    gray_s1 = gray_s1*(255/double(max(gray_s1(:))));
                    gray_s2 = gray_s2*(255/double(max(gray_s2(:))));
                    gray_cat = cat(1,gray_b,gray_s1,gray_s2);
		    im_b = coor_mat_mf.fish_in_vid.b{1}{1}{n};
                    im_s1 = coor_mat_mf.fish_in_vid.s1{1}{1}{n};
                    im_s2 = coor_mat_mf.fish_in_vid.s2{1}{1}{n};
                    buffer_x = round((imageSizeX - size(im_b,2))/2)-1;
                    buffer_y = round((imageSizeY - size(im_b,1))/2)-1;
                    im_b_full = zeros(imageSizeY,imageSizeX,'uint8');
                    im_b_full = imnoise(im_b_full, 'gaussian', 0,  0.00007*(rand+1));
                    im_s1_full = im_b_full;
                    im_s1_full = imnoise(im_s1_full, 'gaussian', 0,  0.00007*(rand+1));
                    im_s2_full = im_b_full;
                    im_s2_full = imnoise(im_s2_full, 'gaussian', 0,  0.00007*(rand+1));
                    im_b_full(buffer_y+1:buffer_y+size(im_b,1), buffer_x+1:buffer_x+size(im_b,2)) = im_b;
                    im_b_full = im_b_full*(255/double(max(im_b_full(:))));
                    buffer_x = round((imageSizeX - size(im_s1,2))/2);
                    buffer_y = round((imageSizeY - size(im_s1,1))/2);
                    im_s1_full(buffer_y+1:buffer_y+size(im_s1,1), buffer_x+1:buffer_x+size(im_s1,2)) = im_s1;
                    im_s1_full = im_s1_full*(255/double(max(im_s1_full(:))));

                    buffer_x = round((imageSizeX - size(im_s2,2))/2);
                    buffer_y = round((imageSizeY - size(im_s2,1))/2);
                    im_s2_full(buffer_y+1:buffer_y+size(im_s2,1), buffer_x+1:buffer_x+size(im_s2,2)) = im_s2;
                    im_s2_full = im_s2_full*(255/double(max(im_s2_full(:))));
                    im = cat(1,im_b_full,im_s1_full,im_s2_full);
		    rgb_im = zeros([size(gray_cat),3],'uint8');
		    rgb_im(:,:,1) = im - gray_cat;
		    rgb_im(:,:,2) = gray_cat - im;
		    imwrite(rgb_im, ['physical_model_inspection/im.png'])
                    idx = idx + 1;
            end
        end
    end
end
% csvwrite('annotations_3D.csv', crop_coor_mat)
