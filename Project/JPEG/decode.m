% Inverse zigzag function
function block = inverse_zigzag(zz, rows, cols)
    % Converts a zigzag-ordered 1D array back to an 8x8 matrix
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
    block = zeros(rows, cols); % Initialize empty matrix
    block(idx) = zz; % Assign zigzag values
end

function decoded_image = jpeg_decode(encoded_file)
    % JPEG Decoding Function
    % encoded_file: Path to the file with encoded data
    % Returns the reconstructed image
    
    % Load encoded data
    load(encoded_file, 'encoded_data', 'q_matrix', 'h', 'w');
    
    % Initialize variables
    decoded_image = zeros(h, w);
    dc_prev = 0; % Previous DC coefficient
    
    % Process encoded blocks
    idx = 1;
    for i = 1:8:h
        for j = 1:8:w
            % Decode DC coefficient
            [dc_diff, idx] = huffman_decode_dc(encoded_data, idx);
            dc_curr = dc_diff + dc_prev;
            dc_prev = dc_curr;
            
            % Decode AC coefficients
            [ac_coeffs, idx] = run_length_decode(encoded_data, idx);
            
            % Reconstruct quantized block
            zz = [dc_curr; ac_coeffs]; % Combine DC and AC coefficients
            quantized = inverse_zigzag(zz, 8, 8);
            
            % De-quantize
            dct_block = quantized .* q_matrix;
            
            % Apply inverse DCT
            block = idct2(dct_block);
            
            % Restore intensity values
            decoded_image(i:i+7, j:j+7) = block + 128;
        end
    end
end
