% Author: Aniket Ravan
% Generates training dataset for side views 
% Last modified: 22nd of May, 2022

load('../proj_params_101019_corrected')
load('../lut_b_tail')
load('../lut_s_tail')
path{1} = '../results_all_er';
% path{2} = '../results_fs';
% path{3} = '../results_ob';
idx = 0;
x_complete = [];
swimtype = 'er';
imageSizeX = 141; imageSizeY = 141;
for path_idx = 1
    coor_mf_mats = dir(path{path_idx});
    for z = 1:10%(length(coor_mf_mats)) - 2
        coor_mf_matname = coor_mf_mats(z+2).name;
        coor_mat_mf = importdata([path{path_idx} '/' coor_mf_matname]);
        [num2str(z),' of ',num2str(length(coor_mf_mats))]
	for i = 1:length(coor_mat_mf.x_all)
            x_all_mf = coor_mat_mf.x_all{i};
            nswimbouts = length(x_all_mf);
            fishlen = coor_mat_mf.fishlen_all(1);
            nframes = length(x_all_mf);
            seglen = fishlen*0.09;
            
            for n = 20:24%nframes
                %if (n/nframes > 0.15 && n/nframes < 0.75 && path_idx == 1)
                
		n_iterations = 2;
                %elseif (n/nframes > 0.15 && n/nframes < 0.75 && path_idx > 1)
                %    n_iterations = 17;
                %else
                    %continue
                %end
                for iter = 1:n_iterations
                    % model fish image
                    x = x_all_mf(n,:);
                    x(4) = rand*2*pi;
                    x(13) = normrnd(0,0.3)*pi/2;                                       
                    [gray_b,gray_s1,gray_s2,crop_b,crop_s1,crop_s2,c_b,c_s1,c_s2,eye_b,eye_s1,eye_s2] = ...
                        return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
                    coor_3d = return_coor_3d(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY);
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
                    imwrite(im,['test//images/im_',num2str(idx),'.png'])                  
%                     imwrite(im_tail,['training_side_tail\im_',num2str(idx),'_',num2str(n),'.png'])
                    crop_coor = [crop_b, crop_s1, crop_s2];
                    pose = [c_b, c_s1, c_s2];
		    c_b = c_b + 1; c_s1 = c_s1 + 1; c_s2 = c_s2 + 1;
                    %save(['../training_data_3D_pose_new/annotations_220612_pose/pose_ann_',num2str(idx),'.mat'],  'pose');
                    %save(['../training_data_3D_pose_new/annotations_220612_crop_coor/crop_coor_ann_',num2str(idx),'.mat'],  'crop_coor');
		    save(['test/pose/pose_',num2str(idx),'.mat'], 'pose');
		    save(['test/crop/crop_',num2str(idx),'.mat'], 'crop_coor');
		    save(['test/coor_3d/coor_3d_',num2str(idx),'.mat'], 'coor_3d');
		    imshow(gray_b)
		    hold on 
		    plot(c_b(1,:), c_b(2,:), 'Marker', 'o', 'LineStyle', 'None', 'MarkerFaceColor', 'None', 'MarkerSize', 0.8)
		    hold off
		    saveas(gcf,['test/b_',num2str(idx),'.png'])
		    imshow(gray_s1)
		    hold on 
		    plot(c_s1(1,:), c_s1(2,:), 'Marker', 'o', 'LineStyle', 'None', 'MarkerFaceColor', 'None', 'MarkerSize', 0.8)
		    hold off
		    saveas(gcf, ['test/s1_',num2str(idx),'.png'])
		    imshow(gray_s2)
		    hold on 
	    	    plot(c_s2(1,:), c_s2(2,:), 'Marker', 'o', 'LineStyle', 'None', 'MarkerFaceColor', 'None', 'MarkerSize', 0.8)
	            hold off
		    saveas(gcf, ['test/s2_',num2str(idx),'.png'])
		    display(['iteration ',num2str(iter)])
		    idx = idx + 1;
                end
            end
        end
    end
end
display('Done')
