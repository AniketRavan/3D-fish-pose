% build side view image from look up table
% This was originally calc_diff_s_lut_new_real_cpu
function [graymodel] = view_s_lut_new_real_cpu(crop_coor, pt,lut_tail,projection,imageSizeX,imageSizeY)
% Find the coefficients of the line that defines the refracted ray


vec_pt = pt(:,2:10) - pt(:,1:9);
segslen = sqrt(sum(vec_pt.*vec_pt,1));
segslen = repmat(segslen,2,1);
vec_pt_unit = vec_pt./segslen;
theta = atan2(vec_pt_unit(2,:),vec_pt_unit(1,:));

% shift pts to the cropped images
pt(1,:) = pt(1,:) - crop_coor(3) + 1;
pt(2,:) = pt(2,:) - crop_coor(1) + 1;

imblank = zeros(imageSizeY,imageSizeX,'uint8');
imblank_cpu = zeros(imageSizeY,imageSizeX,'uint8');
bodypix = imblank_cpu;


headpix = uint8(projection/1.8)*5.2;

% tail
size_lut = 15;
size_half = (size_lut+1)/2;
seglen = segslen;
seglen(seglen < 0.2) = 0.1;
seglen(seglen > 7.9) = 8;
seglenidx = round(seglen/0.2);
coor_t = floor(pt);
dt = floor((pt - coor_t)*5) + 1;
at = mod(floor(theta*90/pi),180) + 1;
%[coor_t,dt,at,seglenidx] = gather(coor_t,dt,at,seglenidx);
for ni = 1:7
    n = ni+2;
    tailpix = imblank;
    tail_model = gen_lut_s_tail(ni, seglenidx(n), dt(1,n), dt(2,n), at(n));
    tailpix(max(1,coor_t(2,n)-(size_half-1)):min(imageSizeY,coor_t(2,n)+(size_half-1)),...
        max(1,coor_t(1,n)-(size_half-1)):min(imageSizeX,coor_t(1,n)+(size_half-1))) =...
        tail_model(max((size_half+1)-coor_t(2,n),1):min(imageSizeY-coor_t(2,n)+size_half,size_lut),...
        max((size_half+1)-coor_t(1,n),1):min(imageSizeX-coor_t(1,n)+size_half,size_lut));
    %lut_tail{ni,seglenidx(n),dt(1,n),dt(2,n),at(n)}(max((size_half+1)-coor_t(2,n),1):min(imageSizeY-coor_t(2,n)+size_half,size_lut),...
        %max((size_half+1)-coor_t(1,n),1):min(imageSizeX-coor_t(1,n)+size_half,size_lut));
    bodypix = max(bodypix, tailpix);
end

graymodel = max(headpix,(normrnd(0.8,0.05))*bodypix);
end
