clear;
decoder('abc.jpeg', 50);

function decoder(encoded_file, quality_factor)
    % Open the binary file for reading
    fid = fopen(encoded_file, 'rb');
    if fid == -1
        error('Unable to open the binary file.');
    end

    % Read original dimensions
    original_height = fread(fid, 1, 'uint32');
    original_width = fread(fid, 1, 'uint32');

    % Read number of Huffman codes
    num_codes = fread(fid, 1, 'uint32');
    huffman_codes = containers.Map();

    % Read each Huffman code
    for i = 1:num_codes
        symbol = fread(fid, 1, 'int32'); % Symbol
        code_length = fread(fid, 1, 'uint32'); % Code length
        code_bits = fread(fid, code_length, 'ubit1')'; % Read code bits
        code = char(code_bits + '0'); % Convert bits to '0'/'1' characters
        huffman_codes(code) = symbol; % Map code to symbol
    end

    % Read encoded data length and content
    encoded_data_length = fread(fid, 1, 'uint32');
    encoded_data_bits = fread(fid, encoded_data_length, 'ubit1')'; % Read encoded data bits
    encoded_data = char(encoded_data_bits + '0'); % Convert bits to '0'/'1' characters

    fclose(fid);

    % Decode the Huffman-encoded data
    decoded_data = decodeHuffman(encoded_data, huffman_codes);

    % Define the JPEG quantization matrix
    Q = [16 11 10 16 24 40 51 61;
         12 12 14 19 26 58 60 55;
         14 13 16 24 40 57 69 56;
         14 17 22 29 51 87 80 62;
         18 22 37 56 68 109 103 77;
         24 35 55 64 81 104 113 92;
         49 64 78 87 103 121 120 101;
         72 92 95 98 112 100 103 99];

    % Adjust quantization matrix based on quality factor
    Q = Q * (50 / quality_factor);
    Q(Q == 0) = 1; % Avoid division by zero

    % Calculate padded dimensions
    pad_height = mod(8 - mod(original_height, 8), 8);
    pad_width = mod(8 - mod(original_width, 8), 8);
    padded_height = original_height + pad_height;
    padded_width = original_width + pad_width;
    rle_index = 1;
    compressed_image = zeros(padded_height, padded_width);

    % Reconstruct compressed_image from decoded RLE data
    for i = 1:8:padded_height
        for j = 1:8:padded_width
            [zigzag_coeffs, rle_index] = runLengthDecode(decoded_data, rle_index);
            
            % Reconstruct the quantized block from zigzag coefficients
            quantized_block = inverseZigzagTransform(zigzag_coeffs, [8, 8]);
            
            compressed_image(i:i+7, j:j+7) = quantized_block;
        end
    end

    % Perform dequantization and inverse DCT
    decompressed_image = zeros(size(compressed_image));
    for i = 1:8:size(compressed_image, 1)
        for j = 1:8:size(compressed_image, 2)
            quantized_block = compressed_image(i:i+7, j:j+7);
            dequantized_block = quantized_block .* Q;
            decompressed_block = idct2(dequantized_block);
            decompressed_image(i:i+7, j:j+7) = decompressed_block;
        end
    end

    % Shift pixel values back by adding 128
    decompressed_image = decompressed_image + 128;

    % Clamp values to the [0, 255] range
    decompressed_image = min(max(decompressed_image, 0), 255);
    decompressed_image = uint8(decompressed_image);

    % Crop to original dimensions
    decompressed_image = decompressed_image(1:original_height, 1:original_width);

    % Display the reconstructed image
    figure;
    imshow(decompressed_image);
    title('Reconstructed Image');
end

function decoded_data = decodeHuffman(encoded_data, huffman_codes)
    % Create a reverse mapping of codes to symbols
    reverse_codes = containers.Map();
    code_keys = keys(huffman_codes);
    for i = 1:length(code_keys)
        code = code_keys{i}; % Huffman code (string)
        symbol = huffman_codes(code); % Corresponding symbol (numeric)
        reverse_codes(code) = symbol; % Store as key-value pair
    end

    % Decode the encoded data
    decoded_data = [];
    buffer = '';
    for i = 1:length(encoded_data)
        buffer = [buffer encoded_data(i)];
        if isKey(reverse_codes, buffer)
            decoded_data = [decoded_data reverse_codes(buffer)];
            buffer = ''; % Reset buffer after successful decode
        end
    end

    % Check for incomplete decoding
    if ~isempty(buffer)
        error('Decoding failed: Remaining buffer contains unprocessed bits.');
    end
end

function zigzag = inverseZigzagTransform(data, dims)
    % Inverse Zigzag transformation to reconstruct an 8x8 block
    idx = [1,     9,    17,    25,    33,    41,    49,  57, ...
           2,    10,    18,    26,    34,    42,    50,    58, ...
           3,    11,    19,    27,    35,    43,    51,    59, ...
           4,    12,    20,    28,    36,    44,    52,    60, ...
           5,    13,    21,    29,    37,    45,    53,    61, ...
           6,    14,    22,    30,    38,    46,    54,    62, ...
           7,    15,    23,    31,    39,    47,    55,    63, ...
           8,    16,    24,    32,    40,    48,    56,    64];
       
    % Initialize a linear array
    zigzag_linear = zeros(1, 64);
    
    % Assign data based on zigzag indices
    zigzag_linear(idx) = data;
    
    % Reshape to the specified dimensions (8x8)
    zigzag = reshape(zigzag_linear, dims);
end

function [decoded, new_idx] = runLengthDecode(encoded_data, start_idx)
    decoded = [];
    idx = start_idx;
    
    while length(decoded) < 64 && idx <= length(encoded_data)-1
        symbol = encoded_data(idx);
        count = encoded_data(idx+1);
        
        % Determine how many symbols to add without exceeding 64
        to_add = min(count, 64 - length(decoded));
        decoded = [decoded, repmat(symbol, 1, to_add)];
        
        idx = idx + 2; % Move to the next [symbol, count] pair
    end
    
    % Ensure exactly 64 coefficients are returned
    decoded = decoded(1:64);
    new_idx = idx;
end
