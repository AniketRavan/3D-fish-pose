function centroid = return_centroid_b(im)

thresh_npix_lower = 135;
thresh_npix_upper = 800;
imblurred = imgaussfilt(im,1);
scaling_factor1 = 0.42;
flag = 0;
while flag == 0
    thresh_bw = graythresh(imblurred(imblurred>20))*scaling_factor1;
    %     mean_pix_val = mean(im(im>20));
    %     thresh_bw = 0.002 * mean_pix_val;
    %     imblurred = imgaussfilt(im,1);
    bw = imbinarize(imblurred,thresh_bw);
    bw = imerode(bw,ones(2));
    CC = bwconncomp(bw);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    numPixels(numPixels>thresh_npix_upper) = 0;
    if (max(numPixels) > 145 || scaling_factor1 < 0.25)
        flag = 1;
    end
    scaling_factor1 = scaling_factor1 - 0.03;
end
flag = 0;
scaling_factor2 = 0.18;
while flag == 0
    thresh_bw = graythresh(imblurred(imblurred>20))*scaling_factor2;
    bw = imbinarize(imblurred,thresh_bw);
    bw = imerode(bw,ones(2));
    CC = bwconncomp(bw);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    numPixels(numPixels>thresh_npix_upper) = 0;
    if (max(numPixels) > 145 || scaling_factor2 > 0.4)
        flag = 1;
    end
    scaling_factor2 = scaling_factor2 + 0.03;
end
mean_scaling_factor = (2*scaling_factor1 + 3*scaling_factor2)/5;
thresh_bw = graythresh(imblurred(imblurred>20))*mean_scaling_factor;
bw = imbinarize(imblurred,thresh_bw);
bw = imerode(bw,ones(2));
bw = bwareaopen(bw,thresh_npix_lower);
centroid = regionprops(bw, 'centroid');
centroid = centroid.Centroid;