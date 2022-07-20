% build bottom view image from look up table
% This was originally calc_diff_s_lut_new_real_cpu

function [graymodel] = view_b_lut_new_real_cpu(crop_coor,pt,lut_tail,projection,imageSizeX,imageSizeY)
%Find the coefficients of the line a,r that defines the refracted ray


vec_pt = pt(:,2:10) - pt(:,1:9);
segslen = sqrt(sum(vec_pt.*vec_pt,1));
segslen = repmat(segslen,2,1);
vec_pt_unit = vec_pt./segslen;
theta = atan2(vec_pt_unit(2,:),vec_pt_unit(1,:));

% shift pts to the cropped images

pt(1,:) = pt(1,:) - crop_coor(3) + 1;
pt(2,:) = pt(2,:) - crop_coor(1) + 1;

imblank = zeros(imageSizeY,imageSizeX,'uint8');
bodypix = imblank;

headpix = uint8(projection/2)*5.2;
size_lut = 19;
size_half = (size_lut+1)/2;

% tail
coor_t = floor(pt);
dt = floor((pt - coor_t)*5) + 1;
at = mod(floor(theta*180/pi),360)+1;
seglen = segslen;
seglen(seglen < 3.3) = 3.2;
seglen(seglen > 10.5) = 10.6;
seglenidx = round((seglen - 5)/0.2);


for ni = 1:7
    n = ni+2;
    tailpix = imblank;
    tail_model = gen_lut_b_tail(ni, seglenidx(n), dt(1,n), dt(2,n), at(n));
    tailpix(max(1,coor_t(2,n)-(size_half-1)):min(imageSizeY,coor_t(2,n)+(size_half-1)),...
        max(1,coor_t(1,n)-(size_half-1)):min(imageSizeX,coor_t(1,n)+(size_half-1))) =...
        tail_model(max((size_half+1)-coor_t(2,n),1):min(imageSizeY-coor_t(2,n)+size_half,size_lut),...
        max((size_half+1)-coor_t(1,n),1):min(imageSizeX-coor_t(1,n)+size_half,size_lut));
        
    %lut_tail{ni,seglenidx(n),dt(1,n),dt(2,n),at(n)}(max((size_half+1)-coor_t(2,n),1):min(imageSizeY-coor_t(2,n)+size_half,size_lut),...
        %max((size_half+1)-coor_t(1,n),1):min(imageSizeX-coor_t(1,n)+size_half,size_lut));
    bodypix = max(bodypix, tailpix);
end

graymodel = max(headpix,(normrnd(0.6,0.08))*bodypix);

end
