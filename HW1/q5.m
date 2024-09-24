J1 = double(imread('T1.jpg'));
J2 = double(imread('T2.jpg'));

theta = 28.5;
eps = 1e-20;

% Rotate the image J2 by theta degrees anti-clockwise
J3 = imrotate(J2, theta, 'bilinear', 'crop');

% Set the value of unoccupied pixels to 0
J3(J3 == 0) = 0;

theta_range = -45:1:45;
ncc_values = zeros(size(theta_range));
je_values = zeros(size(theta_range));
qmi_values = zeros(size(theta_range));

for idx = 1:length(theta_range)
    angle = theta_range(idx);
    J4 = imrotate(J3, angle, 'bilinear', 'crop');
    J4(J4 == 0) = 0;
    
    % 1. Compute the normalized cross-correlation (NCC)
    v1 = (J1(:) - mean(J1(:))) / norm(J1(:) - mean(J1(:)));
    v4 = (J4(:) - mean(J4(:))) / norm(J4(:) - mean(J4(:)));
    ncc_values(idx) = abs(dot(v1, v4));

    % 2. Compute the joint entropy (JE)
    je_prob = joint_histogram(J1, J4, 1); 
    je_prob = je_prob / sum(je_prob(:)); 
    je_values(idx) = -sum(je_prob(:) .* log(je_prob(:) + eps));

    % 3. Compute the quadratic mutual information (QMI)
    joint_hist = joint_histogram(J1, J4, 10); 
    joint_prob = joint_hist / sum(joint_hist(:));
    i1_hist = sum(joint_hist, 2) / sum(joint_hist(:));
    i4_hist = sum(joint_hist, 1) / sum(joint_hist(:));
    

    difference = joint_prob - (i1_hist * i4_hist); 
    qmi_values(idx) = sum(difference(:).^2); 


end

% Find the optimal rotation for each measure
[~, idx_ncc_max] = max(ncc_values);
[~, idx_je_min] = min(je_values);
[~, idx_qmi_min] = max(qmi_values);

optimal_rotation_ncc = theta_range(idx_ncc_max);
optimal_rotation_je = theta_range(idx_je_min);
optimal_rotation_qmi = theta_range(idx_qmi_min);

figure;
% First subplot: Plot NCC values
subplot(1, 2, 1);
imshow(J1/255);
title('Image 1');
subplot(1, 2, 2);
imshow(J3/255);
title('Rotated Image 2');
% Display the optimal rotations
disp(['Optimal rotation (NCC): ', num2str(optimal_rotation_ncc)]);
disp(['Optimal rotation (JE): ', num2str(optimal_rotation_je)]);
disp(['Optimal rotation (QMI): ', num2str(optimal_rotation_qmi)]);

% Plot results
figure;
% First subplot: Plot NCC values
plot(theta_range, ncc_values, '-o');
xlabel('Theta (degrees)');
ylabel('NCC');
title('Normalized Cross-Correlation vs Theta');

figure;
% Second subplot: Plot JE values
plot(theta_range, je_values, '-o');
xlabel('Theta (degrees)');
ylabel('JE');
title('Joint Entropy vs Theta');

figure;
% Third subplot: Plot QMI values
plot(theta_range, qmi_values, '-o');
xlabel('Theta (degrees)');
ylabel('QMI');
title('Quadratic Mutual Information vs Theta');

%Plot Results
figure;
je_opt = imrotate(J3, optimal_rotation_je, 'bilinear', 'crop');
je_opt(je_opt == 0) = 0;
hm = joint_histogram(J1, je_opt, 10);
% Create a heat map using imagesc
imagesc(hm/sum(hm(:)));
colormap('jet');   
colorbar;        
ax = gca;  
xTicks = ax.XTick;
yTicks = ax.YTick;
ax.XTickLabel = arrayfun(@(x) num2str(x * 10), xTicks, 'UniformOutput', false);
ax.YTickLabel = arrayfun(@(y) num2str(y * 10), yTicks, 'UniformOutput', false);
title('Joint Histogram'); 
xlabel('Image 1 Intensity');               
ylabel('Image 2 (at optimal angle) Intensity');                


% Joint Histogram Function
function joint_hist = joint_histogram(I1, I2, bin_width)
    % Initialize the joint histogram
    max_intensity = 256;
    num_bins = floor(max_intensity / bin_width) + 1;
    joint_hist = zeros(num_bins, num_bins);

    % Compute the joint histogram
    for i = 1:size(I1, 1)
        for j = 1:size(I1, 2)
            bin1 = min(floor(I1(i, j) / bin_width) + 1, num_bins);
            bin2 = min(floor(I2(i, j) / bin_width) + 1, num_bins);
            joint_hist(bin1, bin2) = joint_hist(bin1, bin2) + 1;
        end
    end
end

