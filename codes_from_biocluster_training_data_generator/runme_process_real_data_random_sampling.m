% Author: Aniket Ravan
% Generates real dataset for 3-D pose estimation
% Last modified: 16th of July, 2022
% Randomly samples n poses from the real dataset
addpath('../')
load('proj_params_101019_corrected_new')
load('lut_b_tail')
load('lut_s_tail')
path{1} = '../results_all_er';
% path{3} = '../results_ob';
idx = 0;
data_dir = '../../validation_data_3D_pose_220716';
swimtype = 'er';
date = '220716';
imageSizeX = 141; imageSizeY = 141;
mkdir([data_dir,'/annotations_',date,'_crop_coor'])
mkdir([data_dir,'/images_real'])
mkdir([data_dir,'/images_gray'])
mkdir([data_dir,'/annotations_',date,'_coor_3d'])
display('Created directories')
path_idx = 1; coor_mf_mats = dir(path{path_idx});
for z = 1:length(coor_mf_mats) - 2
	coor_mf_matname = coor_mf_mats(z+2).name;
	coor_mat_mf_cells{z} = importdata([path{path_idx} '/' coor_mf_matname]);
end
while 1
    for z = randi([1,length(coor_mf_mats) - 2]) % 191
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = coor_mat_mf_cells{z};
        [num2str(z),' of ',num2str(length(coor_mf_mats))]
        for i = randi([1,length(coor_mat_mf.fish_in_vid.b)])
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            
            for n = randi([1,nframes]) % 33
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
		[gray_b,gray_s1,gray_s2,Crop_b,Crop_s1,Crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                %gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                %im_tail = imnoise(im_tail, 'gaussian', 0,  0.0007*(rand+1));
                gray_b = imnoise(gray_b, 'gaussian', (rand*3+7)/255,  (rand*10+40)/255^2); %mean in range(7,10), var in range(40,50)
                gray_s1 = imnoise(gray_s1, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2);
                gray_b = gray_b*(255/double(max(gray_b(:))));
                gray_s1 = gray_s1*(255/double(max(gray_s1(:))));
                gray_s2 = gray_s2*(255/double(max(gray_s2(:))));
                
                crop_b(1) = crop_b(1) - buffer_y;
                crop_b(2) = crop_b(1) + imageSizeY - 1;
                crop_b(3) = crop_b(3) - buffer_x;
                crop_b(4) = crop_b(3) + imageSizeX - 1;
                
                buffer_x = round((imageSizeX - size(im_s1,2))/2);
                buffer_y = round((imageSizeY - size(im_s1,1))/2);
                im_s1_full(buffer_y+1:buffer_y+size(im_s1,1), buffer_x+1:buffer_x+size(im_s1,2)) = ...
                    im_s1;
                im_s1_full = im_s1_full*(255/double(max(im_s1_full(:))));
                crop_s1(1) = crop_s1(1) - buffer_y;
                crop_s1(2) = crop_s1(1) + imageSizeY - 1;
                crop_s1(3) = crop_s1(3) - buffer_x;
                crop_s1(4) = crop_s1(3) + imageSizeX - 1;
                
                buffer_x = round((imageSizeX - size(im_s2,2))/2);
                buffer_y = round((imageSizeY - size(im_s2,1))/2);
                im_s2_full(buffer_y+1:buffer_y+size(im_s2,1), buffer_x+1:buffer_x+size(im_s2,2)) = ...
                    im_s2;
                im_s2_full = im_s2_full*(255/double(max(im_s2_full(:))));
                crop_s2(1) = crop_s2(1) - buffer_y;
                crop_s2(2) = crop_s2(1) + imageSizeY - 1;
                crop_s2(3) = crop_s2(3) - buffer_x;
                crop_s2(4) = crop_s2(3) + imageSizeX - 1;
                    
                %[coor_b, coor_s1, coor_s2] = calc_proj_w_refra_cpu(coor_3d, proj_params);
                %coor_b(1,:) = coor_b(1,:) - Crop_b(3) + 1;
                %coor_b(2,:) = coor_b(2,:) - Crop_b(1) + 1;
                %coor_s1(1,:) = coor_s1(1,:) - Crop_s1(3) + 1;
                %coor_s1(2,:) = coor_s1(2,:) - Crop_s1(1) + 1;
                %coor_s2(1,:) = coor_s2(1,:) - Crop_s2(3) + 1;
                %coor_s2(2,:) = coor_s2(2,:) - Crop_s2(1) + 1;
                [im_b_full,im_s1_full,im_s2_full] = resize_image(im_b_full,im_s1_full,im_s2_full,crop_b,crop_s1,crop_s2,Crop_b,Crop_s1,Crop_s2);
                
%                 im_b_full = insertMarker(im_b_full, coor_b(:,1:10)', 'circle','color','red','size',1);
%                 im_b_full = im_b_full(:,:,1);
%                 im_s1_full = insertMarker(im_s1_full, coor_s1(:,1:10)', 'circle','color','red','size',1);
%                 im_s1_full = im_s1_full(:,:,1);
%                 im_s2_full = insertMarker(im_s2_full, coor_s2(:,1:10)', 'circle','color','red','size',1);
%                 im_s2_full = im_s2_full(:,:,1);
                
%                 gray_b = insertMarker(gray_b, coor_b(:,1:10)', 'circle','color','red','size',1);
%                 gray_b = gray_b(:,:,1);
%                 gray_s1 = insertMarker(gray_s1, coor_s1(:,1:10)', 'circle','color','red','size',1);
%                 gray_s1 = gray_s1(:,:,1);
%                 gray_s2 = insertMarker(gray_s2, coor_s2(:,1:10)', 'circle','color','red','size',1);
%                 gray_s2 = gray_s2(:,:,1);
                
                im = cat(1,im_b_full,im_s1_full,im_s2_full);
                   
                im_gray = cat(1,gray_b,gray_s1,gray_s2);
                crop_coor = [Crop_b, Crop_s1, Crop_s2];
                imwrite(im,[data_dir,'/images_real/im_',num2str(idx),'.png'])
		imwrite(im_gray, [data_dir,'/images_gray/im_',num2str(idx),'.png'])
                save([data_dir,'/annotations_',date,'_crop_coor/crop_coor_ann_',num2str(idx),'.mat'],'crop_coor');  
                save([data_dir,'/annotations_',date,'_coor_3d/coor_3d_ann_',num2str(idx),'.mat'],'coor_3d');
		idx = idx + 1
		if (idx == 1000)
			display(['Finished ', num2str(idx)])   
			return
		end	            
            end
        end
    end
end

