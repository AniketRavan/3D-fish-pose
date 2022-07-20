function centroid = return_centroid_s(im)

thresh_npix_lower = 40;

imblurred = imgaussfilt(im,1);
thresh_bw = graythresh(imblurred(imblurred > 20))*0.3;
bw = imbinarize(imblurred,thresh_bw);
bw = imerode(bw,ones(2));
bw = bwareaopen(bw,thresh_npix_lower);

centroid = regionprops(bw, 'centroid');
centroid = centroid.Centroid;