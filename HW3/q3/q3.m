% Load grayscale images
I1 = imread('barbara256.png');
I2 = imread('kodak24.png');

% Convert to double precision for filtering operations
I1 = double(I1);
I2 = double(I2);

% Add zero-mean Gaussian noise with sigma = 5
sigma_noise = 5;
I1_noisy = I1 + sigma_noise * randn(size(I1));
I2_noisy = I2 + sigma_noise * randn(size(I2));

% Define parameters for filters
params = [2 2; 15 3; 3 15];

% ----------------------
% MEAN SHIFT FILTER CODE
% ----------------------

% Mean Shift filter function (using a simple approximation)
function I_filtered = meanShiftFilter(image, sigma_s, sigma_r)
    % Get image dimensions
    [rows, cols] = size(image);
    I_filtered = zeros(rows, cols);

    % Neighborhood window size
    window_size = round(3 * sigma_s);

    % Iterate over each pixel in the image
    for i = 1:rows
        for j = 1:cols
            % Create local window centered at (i,j)
            i_min = max(1, i - window_size);
            i_max = min(rows, i + window_size);
            j_min = max(1, j - window_size);
            j_max = min(cols, j + window_size);

            % Extract local patch
            patch = image(i_min:i_max, j_min:j_max);

            % Compute spatial Gaussian weights
            [X, Y] = meshgrid(j_min:j_max, i_min:i_max);
            dist_sq = (X - j).^2 + (Y - i).^2;
            spatial_weights = exp(-dist_sq / (2 * sigma_s^2));

            % Compute range Gaussian weights (difference in intensity)
            range_weights = exp(-((patch - image(i,j)).^2) / (2 * sigma_r^2));

            % Combined weights
            weights = spatial_weights .* range_weights;

            % Apply mean shift (weighted mean of local neighborhood)
            I_filtered(i,j) = sum(sum(weights .* patch)) / sum(sum(weights));
        end
    end
end

% ----------------------
% APPLY MEAN SHIFT FILTER AND SAVE IMAGES
% ----------------------

for i = 1:size(params, 1)
    sigma_s = params(i, 1);
    sigma_r = params(i, 2);

    % Apply mean shift filter to both noisy images
    I1_ms_filtered = meanShiftFilter(I1_noisy, sigma_s, sigma_r);
    I2_ms_filtered = meanShiftFilter(I2_noisy, sigma_s, sigma_r);

    % Save results for barbara256.png
    imwrite(uint8(I1_noisy), ['barbara256_noisy_sigma_' num2str(sigma_noise) '.png']);
    imwrite(uint8(I1_ms_filtered), ['barbara256_meanshift_sigma_s_' num2str(sigma_s) '_sigma_r_' num2str(sigma_r) '.png']);
    
    % Save results for kodak24.png
    imwrite(uint8(I2_noisy), ['kodak24_noisy_sigma_' num2str(sigma_noise) '.png']);
    imwrite(uint8(I2_ms_filtered), ['kodak24_meanshift_sigma_s_' num2str(sigma_s) '_sigma_r_' num2str(sigma_r) '.png']);
end

% ----------------------
% BILATERAL FILTER CODE
% ----------------------

for i = 1:size(params, 1)
    sigma_s = params(i, 1);
    sigma_r = params(i, 2);

    % Apply bilateral filter to both noisy images
    I1_bilateral_filtered = imbilatfilt(I1_noisy, sigma_r, sigma_s);
    I2_bilateral_filtered = imbilatfilt(I2_noisy, sigma_r, sigma_s);

    % Save results for barbara256.png
    imwrite(uint8(I1_bilateral_filtered), ['barbara256_bilateral_sigma_s_' num2str(sigma_s) '_sigma_r_' num2str(sigma_r) '.png']);
    
    % Save results for kodak24.png
    imwrite(uint8(I2_bilateral_filtered), ['kodak24_bilateral_sigma_s_' num2str(sigma_s) '_sigma_r_' num2str(sigma_r) '.png']);
end

