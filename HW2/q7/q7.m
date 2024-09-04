% Load images barbara_img = imread('barbara256.png');
kodak_img = imread('kodak24.png');

% Convert images to grayscale if they are not already(
      optional, based on your images) if size (barbara_img, 3) ==
    3 barbara_img = rgb2gray(barbara_img);
end if size (kodak_img, 3) == 3 kodak_img = rgb2gray(kodak_img);
end

    % Add Gaussian noise with standard deviation σ = 5 noise_std_5 = 5;
barbara_noisy_5 = imnoise(barbara_img, 'gaussian', 0, (noise_std_5 / 255) ^ 2);
kodak_noisy_5 = imnoise(kodak_img, 'gaussian', 0, (noise_std_5 / 255) ^ 2);

% Add Gaussian noise with standard deviation σ = 10 noise_std_10 = 10;
barbara_noisy_10 =
    imnoise(barbara_img, 'gaussian', 0, (noise_std_10 / 255) ^ 2);
kodak_noisy_10 = imnoise(kodak_img, 'gaussian', 0, (noise_std_10 / 255) ^ 2);

% Bilateral filter parameters params = [ 2, 2; 0.1, 0.1; 3, 15 ];

% Create separate figures for noisy images
figure;
imshow(barbara_noisy_5);
title('Barbara with σ = 5 Noise');
saveas(gcf, 'barbara_noisy_5.png');

figure;
imshow(kodak_noisy_5);
title('Kodak with σ = 5 Noise');
saveas(gcf, 'kodak_noisy_5.png');

figure;
imshow(barbara_noisy_10);
title('Barbara with σ = 10 Noise');
saveas(gcf, 'barbara_noisy_10.png');

figure;
imshow(kodak_noisy_10);
title('Kodak with σ = 10 Noise');
saveas(gcf, 'kodak_noisy_10.png');

% Apply and save bilateral filter results for σ = 5 noise
for i = 1:size(params, 1)
    sigma_s = params(i, 1);
sigma_r = params(i, 2);

% Apply bilateral filter to noisy images with σ = 5 noise barbara_filtered_5 =
    mybilateralfilter(barbara_noisy_5, sigma_r, sigma_s);
kodak_filtered_5 = mybilateralfilter(kodak_noisy_5, sigma_r, sigma_s);

    % Save filtered images for barbara256.png with σ = 5 noise
    figure;
    imshow(barbara_filtered_5);
    title(sprintf('Barbara Filtered: σ_s=%.1f, σ_r=%.1f, Noise σ=5', sigma_s,
                  sigma_r));
    saveas(gcf, sprintf('barbara_filtered_5_%d.png', i));

    % Save filtered images for kodak24.png with σ = 5 noise
    figure;
    imshow(kodak_filtered_5);
    title(sprintf('Kodak Filtered: σ_s=%.1f, σ_r=%.1f, Noise σ=5', sigma_s,
                  sigma_r));
    saveas(gcf, sprintf('kodak_filtered_5_%d.png', i));
end

% Apply and save bilateral filter results for σ = 10 noise
for i = 1:size(params, 1)
    sigma_s = params(i, 1);
sigma_r = params(i, 2);

% Apply bilateral filter to noisy images with σ = 10 noise barbara_filtered_10 =
    mybilateralfilter(barbara_noisy_10, sigma_r, sigma_s);
kodak_filtered_10 = mybilateralfilter(kodak_noisy_10, sigma_r, sigma_s);

    % Save filtered images for barbara256.png with σ = 10 noise
    figure;
    imshow(barbara_filtered_10);
    title(sprintf('Barbara Filtered: σ_s=%.1f, σ_r=%.1f, Noise σ=10', sigma_s,
                  sigma_r));
    saveas(gcf, sprintf('barbara_filtered_10_%d.png', i));

    % Save filtered images for kodak24.png with σ = 10 noise
    figure;
    imshow(kodak_filtered_10);
    title(sprintf('Kodak Filtered: σ_s=%.1f, σ_r=%.1f, Noise σ=10', sigma_s,
                  sigma_r));
    saveas(gcf, sprintf('kodak_filtered_10_%d.png', i));
    end
