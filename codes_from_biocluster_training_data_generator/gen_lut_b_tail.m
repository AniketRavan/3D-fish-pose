% generate look up table for tail

function graymodel = gen_lut_b_tail(n, nseglen, d1, d2, a)

size_lut = 19;
size_half = (size_lut+1)/2;

% size of the balls in the model
random_number = normrnd(0.9,0.1);
ballsize = random_number*[3,2,2,2,2,1.5,1.2,1.2,1];
% thickness of the sticks in the model
thickness = random_number*[7,6,5.5,5,4.5,4,3.5,3];
% brightness of the tail
b_tail = [0.7,0.55,0.45,0.40,0.32,0.28,0.2,0.15];

imageSizeX = size_lut;
imageSizeY = size_lut;
[columnsInImage0, rowsInImage0] = meshgrid(1:imageSizeX, 1:imageSizeY);
imblank = zeros(size_lut,size_lut,'uint8');


radius = ballsize(n+1);
th = thickness(n+1);
bt = b_tail(n);
bt_gradient = b_tail(n+1)/b_tail(n);
p_max = normpdf(0,0,th);
seglen = 5 + 0.2 * nseglen;
centerX = size_half + d1/5;
centerY = size_half + d2/5;
columnsInImage = columnsInImage0;
rowsInImage = rowsInImage0;
ballpix = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius.^2;
ballpix = uint8(ballpix) * 255 * bt  * 0.85;
t = 2*pi*(a-1)/360;
pt = zeros(2,2);
R = [cos(t),-sin(t);sin(t),cos(t)];
vec = R * [seglen;0];
pt(:,1) = [size_half + d1/5; size_half + d2/5];
pt(:,2) = pt(:,1) + vec;
stickpix = imblank;
columnsInImage = columnsInImage0;
rowsInImage = rowsInImage0;
if (pt(1,2) - pt(1,1)) ~= 0
    slope = (pt(2,2) - pt(2,1))/(pt(1,2) - pt(1,1));
    % vectors perpendicular to the line segment
    % th is the thickness of the sticks in the model
    vp = [-slope;1]/norm([-slope;1]);
    % one vertex of the rectangle
    V1 = pt(:,2) - vp * th;
    % two sides of the rectangle
    s1 = 2 * vp * th;
    s2 = pt(:,1) - pt(:,2);
    % find the pixels inside the rectangle
    r1 = rowsInImage - V1(2);
    c1 = columnsInImage - V1(1);
    % inner products
    ip1 = r1 * s1(2) + c1 * s1(1);
    ip2 = r1 * s2(2) + c1 * s2(1);
    stickpix_bw = ...
        ip1 > 0 & ip1 < dot(s1,s1) & ip2 > 0 & ip2 < dot(s2,s2);
else
    stickpix_bw = ...
        rowsInImage < max(pt(2,2),pt(2,1)) &...
        rowsInImage > min(pt(2,2),pt(2,1)) &...
        columnsInImage < (pt(1,2) + th) &...
        columnsInImage > (pt(1,2) - th);
end

% the brightness of the points on the stick is a function of its
% distance to the segment
[ys,xs] = ind2sub(size(stickpix_bw),find(stickpix_bw));
px = pt(1,2) - pt(1,1);
py = pt(2,2) - pt(2,1);
pp = px*px + py*py;
% the distance between a pixel and the fish backbone
d_radial = zeros(length(ys),1);
% the distance between a pixel and the anterior end of the
% segment (0 < d_axial < 1)
b_axial = zeros(length(ys),1);
for i = 1:length(ys)
    u = ((xs(i) - pt(1,1)) * px + (ys(i) - pt(2,1)) * py) / pp;
    dx = pt(1,1) + u * px - xs(i);
    dy = pt(2,1) + u * py - ys(i);
    d_radial(i) = dx*dx + dy*dy;
    b_axial(i) = 1 - (1 - bt_gradient) * u * 0.9;
end
b_stick = im2uint8((normpdf(d_radial,0,th)/p_max));

for i = 1:length(ys)
    stickpix(ys(i),xs(i)) = b_stick(i)*b_axial(i);
end

stickpix = stickpix * bt;
graymodel = max(ballpix,stickpix);

