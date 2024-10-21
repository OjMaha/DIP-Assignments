% ORL Face Database Mini Face Recognition System
% Set up paths and initialize variables

orl_folder = './ORL'; % Change this to the correct path
num_subjects_orl = 32; % First 32 people
num_train_images_orl = 6; % First 6 images for training
num_test_images_orl = 4; % Remaining 4 images for testing
image_size_orl = [92, 112]; % ORL image dimensions

% Initialize matrices for training and testing data
train_data_orl = [];
test_data_orl = [];

% Loop through each subject in ORL database
for i = 1:num_subjects_orl
    subject_folder = fullfile(orl_folder, sprintf('s%d', i));
    images = dir(fullfile(subject_folder, '*.pgm')); % Get the list of .pgm images

    % Read first 6 images for training
    for j = 1:num_train_images_orl
        img_file = fullfile(subject_folder, images(j).name);
        img = imread(img_file);
        train_data_orl = [train_data_orl; img(:)']; % Flatten the image and add to training data
    end
    
    % Read remaining 4 images for testing
    for j = (num_train_images_orl + 1):min(length(images), num_train_images_orl + num_test_images_orl)
        img_file = fullfile(subject_folder, images(j).name);
        img = imread(img_file);
        test_data_orl = [test_data_orl; img(:)']; % Flatten the image and add to testing data
    end
end

% Mean normalization
train_data_orl = double(train_data_orl);
test_data_orl = double(test_data_orl);
mean_face_orl = mean(train_data_orl, 1);
train_data_orl_centered = train_data_orl - mean_face_orl;
test_data_orl_centered = test_data_orl - mean_face_orl;

% Eigenfaces using eigs or eig
cov_matrix_orl = train_data_orl_centered' * train_data_orl_centered;
[V_orl, D_orl] = eig(cov_matrix_orl);
[~, eig_order_orl] = sort(diag(D_orl), 'descend');
V_orl = V_orl(:, eig_order_orl);

% Project training and testing data onto eigenvectors
train_coeffs_orl = train_data_orl_centered * V_orl;
test_coeffs_orl = test_data_orl_centered * V_orl;

% Recognition rates
k_values_orl = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];
recognition_rate_orl = zeros(1, length(k_values_orl));

for idx = 1:length(k_values_orl)
    k = k_values_orl(idx);
    correct_count = 0;
    
    for i = 1:size(test_coeffs_orl, 1)
        test_coeff = test_coeffs_orl(i, 1:k);
        diffs = sum((train_coeffs_orl(:, 1:k) - test_coeff).^2, 2);
        [~, closest_idx] = min(diffs);
        
        if ceil(closest_idx / num_train_images_orl) == ceil(i / num_test_images_orl)
            correct_count = correct_count + 1;
        end
    end
    
    recognition_rate_orl(idx) = (correct_count / size(test_coeffs_orl, 1)) * 100;
end


% Create a figure for recognition rates for ORL
figure;

% Plot with enhancements
plot(k_values_orl, recognition_rate_orl, '-o', 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Eigenvectors (k)', 'FontSize', 12);
ylabel('Recognition Rate (%)', 'FontSize', 12);
title('ORL Database Recognition Rate vs Number of Eigenvectors', 'FontSize', 14);
grid on; % Add grid

% Set x-axis and y-axis limits
xlim([min(k_values_orl) max(k_values_orl)]);
ylim([0 100]);

% Customize tick marks and avoid overlap
xticks(k_values_orl);
xtickformat('%.0f'); % Format ticks as integers
set(gca, 'XTickLabelRotation', 45); % Rotate x-axis labels

% Customize y-axis ticks
yticks(0:10:100);

% Add a legend
legend('Recognition Rate', 'Location', 'southeast', 'FontSize', 10);

% Enhance the axes
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on');

% Save the figure
saveas(gcf, 'ORL_Recognition_Rate.png'); % Save as PNG


% SVD Implementation (ORL)
[U_orl, S_orl, V_orl_svd] = svd(train_data_orl_centered, 'econ');

% Projection and recognition using SVD (same as above using V_orl_svd)
train_coeffs_orl_svd = train_data_orl_centered * V_orl_svd;
test_coeffs_orl_svd = test_data_orl_centered * V_orl_svd;

% Recognition loop for SVD (similar to eigenfaces method)
recognition_rate_orl_svd = zeros(1, length(k_values_orl));

for idx = 1:length(k_values_orl)
    k = k_values_orl(idx);
    correct_count = 0;
    
    for i = 1:size(test_coeffs_orl_svd, 1)
        test_coeff = test_coeffs_orl_svd(i, 1:k);
        diffs = sum((train_coeffs_orl_svd(:, 1:k) - test_coeff).^2, 2);
        [~, closest_idx] = min(diffs);
        
        if ceil(closest_idx / num_train_images_orl) == ceil(i / num_test_images_orl)
            correct_count = correct_count + 1;
        end
    end
    
    recognition_rate_orl_svd(idx) = (correct_count / size(test_coeffs_orl_svd, 1)) * 100;
end

% Create a figure for recognition rates for ORL
figure;

% Plot with enhancements
plot(k_values_orl, recognition_rate_orl, '-o', 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'b');
hold on; % Keep the current plot

% Add a smoothed line (optional)
y = recognition_rate_orl; % Use the recognition rates
x = k_values_orl; % Use the k values
p = polyfit(x, y, 3); % Fit a polynomial of degree 3
xx = linspace(min(x), max(x), 100); % Create x values for smooth line
yy = polyval(p, xx); % Evaluate the polynomial at the new x values
plot(xx, yy, 'k--', 'LineWidth', 1.5); % Plot the smooth line

% Set axis labels and title
xlabel('Number of Eigenvectors (k)', 'FontSize', 12);
ylabel('Recognition Rate (%)', 'FontSize', 12);
title('ORL Database Recognition Rate vs Number of Eigenvectors', 'FontSize', 14);

% Add grid
grid on;

% Set x-axis and y-axis limits
xlim([min(k_values_orl) max(k_values_orl)]);
ylim([0 100]);

% Customize tick marks and avoid overlap
xticks(k_values_orl);
xtickformat('%.0f'); % Format ticks as integers
set(gca, 'XTickLabelRotation', 45); % Rotate x-axis labels for readability

% Customize y-axis ticks
yticks(0:10:100);

% Add a legend
legend('Recognition Rate', 'Smoothed Line', 'Location', 'southeast', 'FontSize', 10);

% Enhance the axes
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on');

% Add data labels on points
for i = 1:length(k_values_orl)
    text(k_values_orl(i), recognition_rate_orl(i), sprintf('%.1f%%', recognition_rate_orl(i)), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8);
end


% Yale Face Database Recognition System

yale_folder = './CroppedYale'; % Change this to the correct path
num_subjects_yale = 38; % Number of persons (excluding 14)
num_train_images_yale = 40; % First 40 images for training
num_test_images_yale = 24; % Remaining 24 images for testing
image_size_yale = [192, 168]; % Yale image dimensions

% Initialize matrices for training and testing data
train_data_yale = [];
test_data_yale = [];

% Loop through each subject in Yale database
subfolders_yale = dir(yale_folder);
subfolders_yale = subfolders_yale([subfolders_yale.isdir] & ~ismember({subfolders_yale.name}, {'.', '..', '14'})); % Skip folder '14'

for i = 1:numel(subfolders_yale)
    person_folder = fullfile(yale_folder, subfolders_yale(i).name);
    images = dir(fullfile(person_folder, '*.pgm')); % Get the lis

    % Read first 40 images for training
    for j = 1:num_train_images_yale
        img_file = fullfile(person_folder, images(j).name);
        img = imread(img_file);
        train_data_yale = [train_data_yale; img(:)']; % Flatten the image and add to training data
    end
    
    % Read remaining images for testing
    for j = (num_train_images_yale + 1):min(length(images), num_train_images_yale + num_test_images_yale)
        img_file = fullfile(person_folder, images(j).name);
        img = imread(img_file);
        test_data_yale = [test_data_yale; img(:)']; % Flatten the image and add to testing data
    end
end

% Mean normalization
train_data_yale = double(train_data_yale);
test_data_yale = double(test_data_yale);
mean_face_yale = mean(train_data_yale, 1);
train_data_yale_centered = train_data_yale - mean_face_yale;
test_data_yale_centered = test_data_yale - mean_face_yale;

% Eigenfaces using eigs or eig
cov_matrix_yale = train_data_yale_centered' * train_data_yale_centered;
[V_yale, D_yale] = eig(cov_matrix_yale);
[~, eig_order_yale] = sort(diag(D_yale), 'descend');
V_yale = V_yale(:, eig_order_yale);

% Project training and testing data onto eigenvectors
train_coeffs_yale = train_data_yale_centered * V_yale;
test_coeffs_yale = test_data_yale_centered * V_yale;

% List of k values for recognition
k_values_yale = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];

% Initialize arrays for recognition rates
recognition_rate_yale_all = zeros(1, length(k_values_yale));
recognition_rate_yale_except3 = zeros(1, length(k_values_yale));

% Recognition loop
for idx = 1:length(k_values_yale)
    k = k_values_yale(idx);
    correct_count_all = 0;
    correct_count_except3 = 0;
    
    % Loop through each test image
    for i = 1:size(test_coeffs_yale, 1)
        test_coeff_all = test_coeffs_yale(i, 1:k);
        test_coeff_except3 = test_coeffs_yale(i, 4:k);
        
        % Compute squared differences (all eigencoefficients)
        diffs_all = sum((train_coeffs_yale(:, 1:k) - test_coeff_all).^2, 2);
        % Compute squared differences (excluding first 3 eigencoefficients)
        diffs_except3 = sum((train_coeffs_yale(:, 4:k) - test_coeff_except3).^2, 2);
        
        % Find the closest training image (minimum squared difference)
        [~, closest_train_idx_all] = min(diffs_all);
        [~, closest_train_idx_except3] = min(diffs_except3);
        
        % Check if the predicted person matches the actual person
        if ceil(closest_train_idx_all / num_train_images_yale) == ceil(i / num_test_images_yale)
            correct_count_all = correct_count_all + 1;
        end
        if ceil(closest_train_idx_except3 / num_train_images_yale) == ceil(i / num_test_images_yale)
            correct_count_except3 = correct_count_except3 + 1;
        end
    end
    
    % Calculate recognition rates
    recognition_rate_yale_all(idx) = (correct_count_all / size(test_coeffs_yale, 1)) * 100;
    recognition_rate_yale_except3(idx) = (correct_count_except3 / size(test_coeffs_yale, 1)) * 100;
end

% Create a figure for recognition rates for Yale
figure;

% Plot recognition rates for all eigencoefficients
plot(k_values_yale, recognition_rate_yale_all, '-o', 'MarkerFaceColor', 'b', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'All Eigencoefficients');
hold on;

% Plot recognition rates excluding the first 3 eigencoefficients
plot(k_values_yale, recognition_rate_yale_except3, '-x', 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Excluding First 3 Eigencoefficients');

% Optional: Add smoothed lines
p_all = polyfit(k_values_yale, recognition_rate_yale_all, 3); % Polynomial fit for all eigencoefficients
yy_all = polyval(p_all, k_values_yale); % Evaluate polynomial

p_except3 = polyfit(k_values_yale, recognition_rate_yale_except3, 3); % Polynomial fit for excluding first 3
yy_except3 = polyval(p_except3, k_values_yale); % Evaluate polynomial

% Plot smoothed lines
plot(k_values_yale, yy_all, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Smoothed: All Eigencoefficients');
plot(k_values_yale, yy_except3, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Smoothed: Excluding First 3');

% Set axis labels and title
xlabel('Number of Eigenvectors (k)', 'FontSize', 12);
ylabel('Recognition Rate (%)', 'FontSize', 12);
title('Yale Database Recognition Rate vs Number of Eigenvectors', 'FontSize', 14);

% Add grid
grid on;

% Set x-axis and y-axis limits
xlim([min(k_values_yale) max(k_values_yale)]);
ylim([0 100]);

% Customize tick marks and avoid overlap
xticks(k_values_yale);
xtickformat('%.0f'); % Format ticks as integers
set(gca, 'XTickLabelRotation', 45); % Rotate x-axis labels for readability

% Customize y-axis ticks
yticks(0:10:100);

% Add a legend
legend('show', 'Location', 'southeast', 'FontSize', 10);

% Enhance the axes
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'on');

% Add data labels on points for clarity
for i = 1:length(k_values_yale)
    text(k_values_yale(i), recognition_rate_yale_all(i), sprintf('%.1f%%', recognition_rate_yale_all(i)), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'b');
    text(k_values_yale(i), recognition_rate_yale_except3(i), sprintf('%.1f%%', recognition_rate_yale_except3(i)), ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'r');
end

% Save the figure
saveas(gcf, 'Yale_Recognition_Rate.png'); % Save as PNG

% Face Reconstruction from ORL Database

k_values_reconstruction = [2, 10, 20, 50, 75, 100, 125, 150, 175];
image_idx_to_reconstruct = 1; % Index of image to reconstruct (can change as needed)

original_image = reshape(test_data_orl(image_idx_to_reconstruct, :), image_size_orl);

figure;
subplot(3, 3, 1);
imshow(uint8(original_image));
title('Original Image');

for idx = 1:length(k_values_reconstruction)
    k = k_values_reconstruction(idx);
    reconstructed_coeff = test_coeffs_orl(image_idx_to_reconstruct, 1:k) * V_orl(:, 1:k)';
    reconstructed_image = reconstructed_coeff + mean_face_orl;
    reconstructed_image = reshape(reconstructed_image, image_size_orl);
    
    subplot(3, 3, idx + 1);
    imshow(uint8(reconstructed_image));
    title(sprintf('k = %d', k));
end

% Plot 25 Eigenfaces (ORL Database)
figure;
for i = 1:25
    eigenface = reshape(V_orl(:, i), image_size_orl);
    subplot(5, 5, i);
    imshow(mat2gray(eigenface));
    title(sprintf('Eigenface %d', i));
end