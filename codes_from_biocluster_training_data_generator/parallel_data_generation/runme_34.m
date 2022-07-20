% Author: Aniket Ravan
% Generates training dataset for side views
% Last modified: 30th of June, 2022
addpath('../')
load('../proj_params_101019_corrected_new')
load('../lut_b_tail')
load('../lut_s_tail')
path{1} = '../../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 165000;
data_dir = '../../training_data_3D_pose_shifted/'
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
date = '220629';
mkdir([data_dir,'annotations_',date,'_pose'])
mkdir([data_dir,'annotations_',date,'_crop_coor'])
mkdir([data_dir,'annotations_',date,'_eye_coor'])
mkdir([data_dir,'images'])
mkdir([data_dir,'annotations_',date,'_coor_3d'])
path_idx = 1
coor_mf_mats = dir(path{path_idx});
while 1
    for z = 1:round(((length(coor_mf_mats)) - 2))
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf_vec{z} = importdata([path{path_idx} '/' coor_mf_matname]);
    end
    
    for z = randi([1, length(coor_mf_mats) - 2])
        %coor_mf_matname = coor_mf_mats(z+2).name;
        %coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        coor_mat_mf = coor_mat_mf_vec{z};
        for i = randi([1,length(coor_mat_mf.x_all)])
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            for n = randi([1,nframes])
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
                    if (iter > 1)
                        x(1) = (rand-0.5)*40 + 0;
                        x(2) = (rand-0.5)*40 + 0;
                        x(3) = (rand-0.5)*40 + 70;
                    end
                    x(4) = rand*2*pi;
                    x(13) = normrnd(0,0.3)*pi/2;
                    [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                    eye_coor = [eye_b,eye_s1,eye_s2];
                    gray_b = imgaussfilt(gray_b); gray_s1 = imgaussfilt(gray_s1); gray_s2 = imgaussfilt(gray_s2);
                    %gray_b = imnoise(gray_b, 'gaussian', (rand*3+7)/255,  (rand*10+40)/255^2); %mean in range(7,10), var in range(40,50)
                    %gray_s1 = imnoise(gray_s1, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    %gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    %                     im_tail = imnoise(im_tail, 'gaussian', 0,  0.0007*(rand+1));
                    gray_b = imnoise(gray_b, 'gaussian', (rand*3+7)/255,  (rand*10+40)/255^2); %mean in range(7,10), var in range(40,50)
                    gray_s1 = imnoise(gray_s1, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2);
                    
                    % Gray_b
                    
                    gray_b = gray_b*(255/double(max(gray_b(:))));
                    gray_s1 = gray_s1*(255/double(max(gray_s1(:))));
                    gray_s2 = gray_s2*(255/double(max(gray_s2(:))));
                    im = cat(1,gray_b,gray_s1,gray_s2);
                    %                     im_tail = im_tail*(255/double(max(gray_b(:))));
                    imwrite(im,[data_dir,'images/im_',num2str(idx),'.png'])
                    %                     imwrite(im_tail,['training_side_tail\im_',num2str(idx),'_',num2str(n),'.png'])
                    crop_coor = [crop_b, crop_s1, crop_s2];
                    pose = [c_b, c_s1, c_s2];
                    if (isempty(eye_coor) || isempty(coor_3d) || isempty(pose))
                        display(x)
                        display(coor_mf_matname)
                    end
                    save([data_dir,'annotations_',date,'_coor_3d/coor_3d_',num2str(idx),'.mat'],  'coor_3d');
                    save([data_dir,'annotations_',date,'_pose/pose_ann_',num2str(idx),'.mat'],  'pose');
                    save([data_dir,'annotations_',date,'_crop_coor/crop_coor_ann_',num2str(idx),'.mat'],  'crop_coor');
                    save([data_dir,'annotations_',date,'_eye_coor/eye_coor_ann_',num2str(idx),'.mat'],  'eye_coor');
                    idx = idx + 1;
                    if (mod(idx,1) == 5000); display(idx); end
                    if idx == 170000
                        display('Finished ' + num2str(170000)  )
                        return
                    end
                end
            end
        end
    end
end
% csvwrite('annotations_3D.csv', crop_coor_mat)
display('Finished running')
