im1 = double(imread('goi1.jpg'));
im2 = double(imread('goi2_downsampled.jpg'));


i1 = ones(3, 12);  
i2 = ones(3, 12);  


% Manually select 12 pairs of corresponding points
for i = 1:12
    % Select point on the first image
    figure(1); imshow(im1/255);
    [i1(1,i), i1(2,i)] = ginput(1);
    
    % Select corresponding point on the second image
    figure(2); imshow(im2/255);
    [i2(1,i), i2(2,i)]= ginput(1);
end

% T = zeros(3,3);
% T(3,3) = 1;
T = i2*pinv(i1);

disp(T)

[r,c] = size(im1);

warp_linear = zeros(size(im2));

for x = 1:c
    for y = 1:r

        revwarp = T\ [x; y; 1];
        x_inv = revwarp(1);
        y_inv = revwarp(2);

         % Perform inverse
         nearest_x = round(x_inv);
         nearest_y = round(y_inv);
 
         % Checking bounds
         if nearest_x >= 1 && nearest_x <= c && nearest_y >= 1 && nearest_y <= r
            warp_linear(y, x) = im1(nearest_y, nearest_x); 
         end
    end
end


warp_bilinear = zeros(size(im2));

for x = 1:c
    for y = 1:r

        revwarp = T\ [x; y; 1];
        x_inv = revwarp(1);
        y_inv = revwarp(2);

         % Performing inverse
         xd = floor(x_inv);
         xu = xd+1;
         yd = round(y_inv);
         yu = yd+1;
 
         % Checking bounds
         if xd >= 1 && xd <= c && yd >= 1 && yd <= r
            warp_bilinear(y, x) = warp_bilinear(y, x) + im1(yd, xd)*(x_inv - xd)*(y_inv - yd); 
         end
         if xd >= 1 && xd <= c && yu >= 1 && yu <= r
            warp_bilinear(y, x) = warp_bilinear(y, x) + im1(yu, xd)*(x_inv - xd)*(yu - y_inv); 
         end
         if xu >= 1 && xu <= c && yd >= 1 && yd <= r
            warp_bilinear(y, x) = warp_bilinear(y, x) + im1(yd, xu)*(xu - x_inv)*(y_inv - yd); 
         end
         if xu >= 1 && xu <= c && yu >= 1 && yu <= r
            warp_bilinear(y, x) = warp_bilinear(y, x) + im1(yu, xu)*(xu - x_inv)*(yu - y_inv); 
         end
    end
end

figure;
imshow(im1/255);
title("Image 1");
figure;
imshow(im2/255);
title("Image 2");
figure;
imshow(warp_linear/255);
title("Nearest Neighbour Interpolation");
figure;
imshow(warp_bilinear/255);
title("Bilinear Interpolation");