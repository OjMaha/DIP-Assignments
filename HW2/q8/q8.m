% Load images
LC1 = imread('LC1.png');
LC2 = imread('LC2.jpg');

% Define the local equalization function
localEqualize = @(block_struct) histeq(block_struct.data);

% Neighborhood sizes to use for local equalization
neighborhoodSizes = [7, 31, 51, 71];

% Perform local histogram equalization for each neighborhood size
for i = 1:length(neighborhoodSizes)
    size = neighborhoodSizes(i);
    
    % Local histogram equalization using blockproc
    LC1_local_eq = blockproc(LC1, [size size], localEqualize);
    LC2_local_eq = blockproc(LC2, [size size], localEqualize);
    
    % Display and save LC1 local histogram equalization
    figure;
    imshow(LC1_local_eq);
    title(['LC1 Local Histogram Equalization (', num2str(size), 'x', num2str(size), ')']);
    % Save the figure to MATLAB Drive
    saveas(gcf, ['LC1_Local_Histogram_Equalization_' num2str(size) 'x' num2str(size) '.png']);
    
    % Display and save LC2 local histogram equalization
    figure;
    imshow(LC2_local_eq);
    title(['LC2 Local Histogram Equalization (', num2str(size), 'x', num2str(size), ')']);
    % Save the figure to MATLAB Drive
    saveas(gcf, ['LC2_Local_Histogram_Equalization_' num2str(size) 'x' num2str(size) '.png']);
end

% Global histogram equalization
LC1_global_eq = histeq(LC1);
LC2_global_eq = histeq(LC2);

% Display and save LC1 global histogram equalization
figure;
imshow(LC1_global_eq);
title('LC1 Global Histogram Equalization');
% Save the figure to MATLAB Drive
saveas(gcf, 'LC1_Global_Histogram_Equalization.png');

% Display and save LC2 global histogram equalization
figure;
imshow(LC2_global_eq);
title('LC2 Global Histogram Equalization');
% Save the figure to MATLAB Drive
saveas(gcf, 'LC2_Global_Histogram_Equalization.png');