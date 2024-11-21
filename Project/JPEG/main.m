% Load grayscale image
input_image = imread('barbara256.png'); % Replace with your image path
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
end

% Define Q=50 quantization matrix
q_matrix_50 = [
    16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99
];

% Encoding
jpeg_encode(input_image, q_matrix_50, 50, 'encoded_data.mat');

% Decoding
decoded_image = jpeg_decode('encoded_data.mat');

% Display results
figure;
subplot(1, 2, 1);
imshow(input_image, []);
title('Original Image');

subplot(1, 2, 2);
imshow(decoded_image, []);
title('Decoded Image');
