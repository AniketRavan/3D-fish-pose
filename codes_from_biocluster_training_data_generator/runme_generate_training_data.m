% Author: Aniket Ravan
% Generates training dataset for side views
% Last modified: 30th of June, 2022
addpath('../')
load('proj_params_101019_corrected_new')
load('lut_b_tail')
load('lut_s_tail')
path{1} = '../training_folder/results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 0;
data_dir = '../training_data_3D_220719/';
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
date = '220719';
mkdir([data_dir,'annotations_',date,'_pose'])
mkdir([data_dir,'annotations_',date,'_crop_coor'])
mkdir([data_dir,'annotations_',date,'_eye_coor'])
mkdir([data_dir,'images']); mkdir([data_dir,'images_channels'])
mkdir([data_dir,'annotations_',date,'_coor_3d'])
path_idx = 1;
coor_mf_mats = dir(path{path_idx});
myStream = RandStream('mlfg6331_64');
RandStream.setGlobalStream(myStream)

x_all_er_3D = importdata('../x_all_er_3D.mat');
while 1
    
    for z = randi([1, length(x_all_er_3D)])
        %coor_mf_matname = coor_mf_mats(z+2).name;
        %coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        %coor_mat_mf = coor_mat_mf_vec{z};
        for i = 1%randi([1,length(coor_mat_mf.x_all)])
            x_all_mf = x_all_er_3D{z}; %coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = normrnd(3.8,0.15); %coor_mat_mf.fishlen_all(1);
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
                        x(1) = (rand-0.5)*40 ;
                        x(2) = (rand-0.5)*40;
                        x(3) = (rand-0.5)*35 + 72.5;
                    end
                    x(4) = rand*2*pi;
                    x(13) = normrnd(0,0.3)*pi/2;
                    [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2,coor_3d] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                    cent_b = [0,0]; cent_s1 = [0,0]; cent_s2 = [0,0];
                    %cent_b = return_centroid_b(gray_b);
                    %cent_s1 = return_centroid_s(gray_s1);
                    %cent_s2 = return_centroid_s(gray_s2); 
                    
                    %Crop_b(1) = crop_b(1) + (cent_b(2) - (imageSizeY+1)/2);
                    %Crop_b(2) = Crop_b(1) + imageSizeY - 1;
                    %Crop_b(3) = crop_b(3) + (cent_b(1) - (imageSizeX+1)/2);
                    %Crop_b(4) = Crop_b(3) + imageSizeX - 1;
                    
                    %Crop_s1(1) = crop_s1(1) + (cent_s1(2) - (imageSizeY+1)/2);
                    %Crop_s1(2) = Crop_s1(1) + imageSizeY - 1;
                    %Crop_s1(3) = crop_s1(3) + (cent_s1(1) - (imageSizeX+1)/2);
                    %Crop_s1(4) = Crop_s1(3) + imageSizeX - 1;
                    
                    %Crop_s2(1) = crop_s2(1) + (cent_s2(2) - (imageSizeY+1)/2);
                    %Crop_s2(2) = Crop_s2(1) + imageSizeY - 1;
                    %Crop_s2(3) = crop_s2(3) + (cent_s2(1) - (imageSizeX+1)/2);
                    %Crop_s2(4) = Crop_s2(3) + imageSizeX - 1;
                    
                    %[gray_b, gray_s1, gray_s2] = resize_image(gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2, ...
   % Crop_b, Crop_s1, Crop_s2);
                    eye_coor = [eye_b,eye_s1,eye_s2];
 		    filter_size = 2*round(rand([1,3])) +  3; sigma = rand([1,3]) + 0.5;
                    gray_b = imgaussfilt(gray_b,sigma(1),'FilterSize',filter_size(1)); gray_s1 = imgaussfilt(gray_s1,sigma(2),'FilterSize',filter_size(2)); gray_s2 = imgaussfilt(gray_s2,sigma(3),'FilterSize',filter_size(3));
                    % Gray_b
                    
                    %gray_b = imnoise(gray_b, 'gaussian', (rand*3+7)/255,  (rand*10+40)/255^2); %mean in range(7,10), var in range(40,50)
                    %gray_s1 = imnoise(gray_s1, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    %gray_s2 = imnoise(gray_s2, 'gaussian', (rand*1+4)/255,  (rand*2+10)/255^2); %mean in range(4,5), var in range(10,12)
                    %                     im_tail = imnoise(im_tail, 'gaussian', 0,  0.0007*(rand+1));
                    
                    
                    gray_b = imnoise(gray_b, 'gaussian', (rand*normrnd(65,15))/255,  (rand*70+20)/255^2); %mean in range(7,10), var in range(40,50)
                    gray_s1 = imnoise(gray_s1, 'gaussian', (rand*normrnd(40,15))/255,  (rand*70+10)/255^2); %mean in range(4,5), var in range(10,12)
                    gray_s2 = imnoise(gray_s2, 'gaussian', (rand*normrnd(40,15))/255,  (rand*70+10)/255^2);
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
                    if (mod(idx,1) == 0); display(idx); end
                    if (idx == 1)
                        display('Finished 500000')
                        return
                    end
                end
            end
        end
    end
end
% csvwrite('annotations_3D.csv', crop_coor_mat)
display('Finished running')