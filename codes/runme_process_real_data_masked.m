% Author: Aniket Ravan
% Generates training dataset for side views
% Last modified: 12th of May, 2022

load('proj_params_101019_corrected_new')
load('../lut_b_tail')
load('../lut_s_tail')
path{1} = '../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 1;
x_complete = [];
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
mkdir('../validation_data_3D_pose_new/crop_coor')
mkdir('../validation_data_3D_pose_new/images')
for path_idx = 1
    coor_mf_mats = dir(path{path_idx});
    for z = 1:(length(coor_mf_mats)) - 2
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        [num2str(z),' of ',num2str(length(coor_mf_mats))]
        for i = 1:length(coor_mat_mf.x_all)
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            se = strel('disk',4);
            for n = 1:nframes
                x = x_all_mf(n,:);
                im_b = coor_mat_mf.fish_in_vid.b{1}{1}{n};
                mask_b = imclose(im2bw(im_b, graythresh(im_b)*0.3),se);
                mask_b = bwareaopen(mask_b,10);
                mask_b = imclearborder(mask_b);
                %mask_b = coor_mat_mf.fish_in_vid.b{1}{4}{n};
                im_b = im_b.*uint8(mask_b);
                im_s1 = coor_mat_mf.fish_in_vid.s1{1}{1}{n};
                mask_s1 = imclose(im2bw(im_s1, graythresh(im_s1)*0.3),se);
                mask_s1 = bwareaopen(mask_s1,10);
                mask_s1 = imclearborder(mask_s1);
                %mask_s1 = coor_mat_mf.fish_in_vid.s1{1}{4}{n};
                im_s1 = im_s1.*uint8(mask_s1);
                im_s2 = coor_mat_mf.fish_in_vid.s2{1}{1}{n};
                mask_s2 = imclose(im2bw(im_s2, graythresh(im_s2)*0.3),se);
                mask_s2 = bwareaopen(mask_s2,10);
                mask_s2 = imclearborder(mask_s2);
                %mask_s2 = coor_mat_mf.fish_in_vid.s2{1}{4}{n};
                im_s2 = im_s2.*uint8(mask_s2);
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
                crop_b = coor_mat_mf.fish_in_vid.b{1}{2}{n};
                crop_s1 = coor_mat_mf.fish_in_vid.s1{1}{2}{n};
                crop_s2 = coor_mat_mf.fish_in_vid.s2{1}{2}{n};
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
                im_b = im_b*(255/double(max(im_b(:))));
%                 noise = im_b(1:25,1:50);
%                 mean_b(idx) = mean(noise(:));
%                 var_b(idx) = var(double(noise(:)));
%                 subplot(3,2,1), histogram(noise(noise > 0))
%                 xlim([0 30])
%                 subplot(3,2,2), imshow(im_b); hold on; rectangle('Position',[1,1,50,20],'Linewidth',2,'EdgeColor','r')
%                 subplot(3,2,3), histogram(noise(noise > 0))
%                 im_s1 = im_s1*(255/double(max(im_s1(:))));
%                 noise = im_s1(1:25,1:50);
%                 mean_s1(idx) = mean(noise(:));
%                 var_s1(idx) = var(double(noise(:)));
%                 xlim([0 30])
%                 subplot(3,2,4), imshow(im_s1); hold on; rectangle('Position',[1,1,50,20],'Linewidth',2,'EdgeColor','r')
%                 im_s2 = im_s2*(255/double(max(im_s2(:))));
%                 noise = im_s2(1:25,1:50);
%                 mean_s2(idx) = mean(noise(:));
%                 var_s2(idx) = var(double(noise(:)));
%                 subplot(3,2,5), histogram(noise(noise > 0))
%                 xlim([0 30])
%                 subplot(3,2,6), imshow(im_s2); hold on; rectangle('Position',[1,1,50,20],'Linewidth',2,'EdgeColor','r')
%                 saveas(gcf, ['histograms_noise_characterization/hist_',num2str(idx),'.png'])
                im = cat(1,im_b_full,im_s1_full,im_s2_full);
                crop_coor = [crop_b, crop_s1, crop_s2];
                imwrite(im,['../validation_data_3D_pose_new/images/im_',num2str(idx),'.png'])
                save(['../validation_data_3D_pose_new/crop_coor/crop_coor_',num2str(idx),'.mat'],'crop_coor');  
                idx = idx + 1;               
            end
        end
    end
end
