function jpeg_encode(input_image, q_matrix_50, quality_factor, output_file)
    % JPEG Encoding Function
    % input_image: Grayscale image to be encoded
    % q_matrix_50: Q=50 quantization matrix
    % quality_factor: Desired quality factor
    % output_file: Output file path to save the encoded data
    
    % Scale quantization matrix
    q_matrix = scale_quantization_matrix(q_matrix_50, quality_factor);
    
    % Prepare image and variables
    image = double(input_image) - 128; % Shift intensity values
    [h, w] = size(image);
    dc_prev = 0; % Previous DC coefficient
    encoded_data = []; % Store encoded data
    
    % Process 8x8 blocks
    for i = 1:8:h
        for j = 1:8:w
            % Extract an 8x8 block
            block = image(i:i+7, j:j+7);
            
            % Apply DCT
            dct_block = dct2(block);
            
            % Quantization
            quantized = round(dct_block ./ q_matrix);
             
            % DC Coefficient Encoding
            dc_curr = quantized(1, 1);
            dc_diff = dc_curr - dc_prev; % Difference with previous DC
            dc_prev = dc_curr;
            dc_encoded = huffman_encode_dc(dc_diff); % Huffman encode DC
            
            % Zigzag order for AC coefficients
            zz = zigzag(quantized);
            ac_coeffs = zz(2:end); % Exclude DC coefficient
            
            % Run-Length Encoding for AC coefficients
            ac_encoded = run_length_encode(ac_coeffs);
            
            % Append encoded data
            encoded_data = [encoded_data; dc_encoded; ac_encoded];
        end
    end
    
    % Save data to file
    save(output_file, 'encoded_data', 'q_matrix', 'h', 'w', '-mat');
end

function qm_scaled = scale_quantization_matrix(qm_50, Q)
    % Scale quantization matrix for a given quality factor Q
    scale = 50 / Q;
    qm_scaled = round(qm_50 * scale);
    qm_scaled(qm_scaled < 1) = 1; % Ensure no element is less than 1
end

% Zigzag scanning function
function zz = zigzag(block)
    % Converts an 8x8 matrix into a zigzag-ordered 1D array
    idx = [
        1  2  6  7 15 16 28 29;
        3  5  8 14 17 27 30 43;
        4  9 13 18 26 31 42 44;
       10 12 19 25 32 41 45 54;
       11 20 24 33 40 46 53 55;
       21 23 34 39 47 52 56 61;
       22 35 38 48 51 57 60 62;
       36 37 49 50 58 59 63 64
    ];
    zz = block(idx); % Use linear indexing to reorder
end