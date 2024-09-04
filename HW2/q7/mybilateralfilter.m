function output = mybilateralfilter(input_image, sigma_r, sigma_s)
    % Convert input image to double for computation
    input_image = double(input_image);

% Define the size of the Gaussian spatial kernel half_window =
    ceil(3 * sigma_s);
[ X, Y ] = meshgrid(-half_window : half_window, -half_window : half_window);

% Gaussian spatial kernel spatial_kernel =
    exp(-(X.^ 2 + Y.^ 2) / (2 * sigma_s ^ 2));

% Initialize the output image[rows, cols] = size(input_image);
output = zeros(rows, cols);

    % Traverse the image with two nested for-loops
    for i = 1:rows
        for j = 1:cols
            % Extract local region
            imin = max(i-half_window, 1);
    imax = min(i + half_window, rows);
    jmin = max(j - half_window, 1);
    jmax = min(j + half_window, cols);

    local_region = input_image(imin : imax, jmin : jmax);

    % Compute Gaussian range kernel based on intensity difference
            intensity_diff = local_region - input_image(i, j);
    range_kernel = exp(-(intensity_diff.^ 2) / (2 * sigma_r ^ 2));

    % Compute bilateral filter response combined_kernel =
        range_kernel.*spatial_kernel((imin
                                      : imax) -
                                         i + half_window + 1,
                                     (jmin
                                      : jmax) -
                                         j + half_window + 1);
    normalization_factor = sum(combined_kernel( :));
    output(i,
           j) = sum(sum(combined_kernel.*local_region)) / normalization_factor;
    end end

        % Convert the output image back to the original type output =
        uint8(output);
    end
