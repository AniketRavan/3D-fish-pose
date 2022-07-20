function im_channels = returned_annotation_channels(im, coor_b, coor_s1, coor_s2, eye_b, eye_s1, eye_s2)

[X,Y] = meshgrid([1:size(im,2)],[1:size(im,1)]);
sigma = 2;
im_channels = zeros(141,141,3*size(coor_b,2) + 6,'uint8');

im_channels(:,:,1) = 250*max(exp(-(1/2)*((X-eye_b(1,1) - 1).^2 + (Y-eye_b(2,1) - 1).^2)/sigma^2),...
    exp(-(1/2)*((X-eye_b(1,2) - 1).^2 + (Y-eye_b(2,2) - 1).^2)/sigma^2));

for i = 1:size(coor_b,2)
    im_channels(:,:,i+1) = 250*exp(-(1/2)*((X-coor_b(1,i) - 1).^2 + (Y-coor_b(2,i) - 1).^2)/sigma^2);
end

im_channels(:,:,12) = 250*max(exp(-(1/2)*((X-eye_s1(1,1) - 1).^2 + (Y-eye_s1(2,1) - 1).^2)/sigma^2),...
    exp(-(1/2)*((X-eye_s1(1,2) - 1).^2 + (Y-eye_s1(2,2) - 1).^2)/sigma^2));

for i = 1:size(coor_s1,2)
    im_channels(:,:,i+12) = 250*exp(-(1/2)*((X-coor_s1(1,i) - 1).^2 + (Y-coor_s1(2,i) - 1).^2)/sigma^2);
end

im_channels(:,:,23) = 250*max(exp(-(1/2)*((X-eye_s2(1,1) - 1).^2 + (Y-eye_s2(2,1) - 1).^2)/sigma^2),...
    exp(-(1/2)*((X-eye_s2(1,2) - 1).^2 + (Y-eye_s2(2,2) - 1).^2)/sigma^2));

for i = 1:size(coor_s2,2)
    im_channels(:,:,i+23) = 250*exp(-(1/2)*((X-coor_s2(1,i) - 1).^2 + (Y-coor_s2(2,i) - 1).^2)/sigma^2);
end