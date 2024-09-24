% Define the image dimensions
N = 201;
image = zeros(N, N);

% Create the image with a central column of 255
image(:, 101) = 255;

% Compute the 2D Fourier transform
F = fft2(image);

% Shift the zero frequency component to the center
F_shifted = fftshift(F);

% Compute the magnitude and take the logarithm
magnitude = abs(F_shifted);
log_magnitude = log(1 + magnitude); % Adding 1 to avoid log(0)

% Plot the result
figure;
imagesc(log_magnitude);
colorbar;
title('Logarithm of Fourier Magnitude');
xlabel('Frequency (u)');
ylabel('Frequency (v)');
axis image; % Set the aspect ratio to be equal
colormap jet; % Choose a colormap for better visualization
