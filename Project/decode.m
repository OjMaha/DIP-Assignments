decoder('compressed_data', 50);

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
    code = char(fread(fid, code_length, 'ubit1')' + '0'); % Read code as binary string
    huffman_codes(code) = symbol; % Map code to symbol
end

% Read encoded data length and content
encoded_data_length = fread(fid, 1, 'uint32');
encoded_data = char(fread(fid, encoded_data_length, 'ubit1')' + '0'); % Read encoded data

fclose(fid);

% Decode the Huffman-encoded data
decoded_data = decodeHuffman(encoded_data, huffman_codes);

% Calculate padded dimensions
padded_height = original_height + mod(8 - mod(original_height, 8), 8);
padded_width = original_width + mod(8 - mod(original_width, 8), 8);
total_elements = padded_height * padded_width;

% Check for dimension mismatch
if length(decoded_data) ~= total_elements
    error('Decoded data size (%d) does not match expected size (%d)', length(decoded_data), total_elements);
end

% Reshape decoded data
compressed_image = reshape(decoded_data, [padded_height, padded_width]);

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

% Decompression: Perform inverse quantization and inverse DCT
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
