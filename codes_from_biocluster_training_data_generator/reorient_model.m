function [X, Y, Z, indices] = reorient_model(model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)
if (isempty(model))
    indices = 1:length(x_c);
else
    indices = find(model > 0);
end

R = rotz(heading)*roty(inclination)*rotx(roll);
new_coor = R*[x_c(indices) - hinge(1); y_c(indices) - hinge(2); z_c(indices) - hinge(3)];
X = new_coor(1,:) + hinge(1) + ref_vec(1); Y = new_coor(2,:) + hinge(2) + ref_vec(2); Z = new_coor(3,:) + hinge(3) + ref_vec(3);

