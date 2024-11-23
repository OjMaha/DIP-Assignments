clear;
input_image = 'kodak24.png';
encoder(input_image, 50, 'abc.jpeg');


function encoder(image_path, quality_factor, output_file)
    % Load the grayscale image
    image = imread(image_path);
    if size(image, 3) == 3
        image = rgb2gray(image); % Convert to grayscale if it's an RGB image
    end
    image = double(image); % Convert to double for processing

    figure;
    imshow(image, []);
    title('Original Image');

    % Store original dimensions
    [original_height, original_width] = size(image);

    % Ensure the image dimensions are multiples of 8
    pad_height = mod(8 - mod(original_height, 8), 8);
    pad_width = mod(8 - mod(original_width, 8), 8);
    if pad_height > 0 || pad_width > 0
        image = padarray(image, [pad_height, pad_width], 'post');
    end
    [height, width] = size(image);

    % Define JPEG quantization matrix for grayscale (luminance)
    Q = [16 11 10 16 24 40 51 61;
         12 12 14 19 26 58 60 55;
         14 13 16 24 40 57 69 56;
         14 17 22 29 51 87 80 62;
         18 22 37 56 68 109 103 77;
         24 35 55 64 81 104 113 92;
         49 64 78 87 103 121 120 101;
         72 92 95 98 112 100 103 99];

    Q = Q * (50 / quality_factor);
    Q(Q == 0) = 1; % Avoid division by zero

    % Shift the image values by -128 and prepare for compression
    sht_image = image - 128;
    compressed_image = zeros(size(sht_image));

    % Compress the image in 8x8 blocks with DCT and quantization
    for i = 1:8:height
        for j = 1:8:width
            block = sht_image(i:i+7, j:j+7);
            dct_block = dct2(block);
            quantized_block = round(dct_block ./ Q);
            compressed_image(i:i+7, j:j+7) = quantized_block;
        end
    end

    % Perform zigzag transformation and RLE on the quantized blocks
    rle_data = [];
    for i = 1:8:height
        for j = 1:8:width
            block = compressed_image(i:i+7, j:j+7);
            zigzag_coeffs = zigzagTransform(block);
            if i == 1
                if j==9
                    %disp(block);
                    % disp(runLengthEncode(zigzag_coeffs));
                end
            end
            rle_data = [rle_data, runLengthEncode(zigzag_coeffs)];
        end
    end

    % Huffman encoding
    [unique_symbols, ~, idx] = unique(rle_data);
    frequencies = accumarray(idx, 1);
    [~, huffman_codes] = huffman_manual(unique_symbols, frequencies);
    encoded_data = encodeData(rle_data, huffman_codes);

    % Save metadata and encoded data to binary file
    fid = fopen(output_file, 'wb');
    if fid == -1
        error('Unable to create the binary file for writing.');
    end

    % Write original dimensions
    fwrite(fid, original_height, 'uint32');
    fwrite(fid, original_width, 'uint32');

    % Write Huffman codes
    num_codes = length(huffman_codes);
    fwrite(fid, num_codes, 'uint32');
    for i = 1:num_codes
        fwrite(fid, huffman_codes(i).symbol, 'int32'); % Write symbol using correct indexing
        fwrite(fid, length(huffman_codes(i).code), 'uint32'); % Write code length
        for bit = huffman_codes(i).code  % Iterate over each character in the Huffman code
            fwrite(fid, bit - '0', 'ubit1'); % Write each bit, converting char '0'/'1' to numeric 0/1
        end
    end

    % Write encoded data length and content
    fwrite(fid, length(encoded_data), 'uint32');
    for bit = encoded_data
        fwrite(fid, bit - '0', 'ubit1'); % Write encoded data as binary
    end

    fclose(fid);
end

function output = zigzagTransform(block)
    % Zigzag order for an 8x8 block
    idx = [1,     9,    17,    25,    33,    41,    49,  57, ...
         2,    10,    18,    26,    34,    42,    50,    58, ...
         3,    11,    19,    27,    35,    43,    51,    59, ...
         4,    12,    20,    28,    36,    44,    52,    60, ...
         5,    13,    21,    29,    37,    45,    53,    61, ...
         6,    14,    22,    30,    38,    46,    54,    62, ...
         7,    15,    23,    31,    39,    47,    55,    63, ...
         8,    16,    24,    32,    40,    48,    56,    64];
    output = block(idx);
end

function [huffman_tree, huffman_codes] = huffman_manual(symbols, frequencies)
    % Step 1: Create the Huffman Tree
    % Create a priority queue (sorted list of nodes)
    nodes = cell(length(symbols), 1);
    for i = 1:length(symbols)
        nodes{i} = struct('symbol', symbols(i), 'freq', frequencies(i), 'left', [], 'right', []);
    end
    
    % Sort nodes by frequency (smallest frequency first)
    nodes = sortNodesByFrequency(nodes);
    
    % Build the Huffman tree
    while length(nodes) > 1
        % Combine the two nodes with the smallest frequencies
        left = nodes{1};
        right = nodes{2};
        
        % Create a new internal node with combined frequency
        new_node = struct('symbol', [], 'freq', left.freq + right.freq, 'left', left, 'right', right);
        
        % Remove the two smallest nodes and add the new node
        nodes = nodes(3:end);
        nodes{end+1} = new_node;
        
        % Sort nodes again by frequency
        nodes = sortNodesByFrequency(nodes);
    end
    
    % The root of the Huffman tree is the only remaining node
    huffman_tree = nodes{1};
    
    % Step 2: Generate Huffman codes from the tree
    huffman_codes = generateHuffmanCodes(huffman_tree, '');
end

% Helper function to sort nodes by frequency
function sorted_nodes = sortNodesByFrequency(nodes)
    % Sort nodes in ascending order of frequency
    [~, idx] = sort(cellfun(@(node) node.freq, nodes));
    sorted_nodes = nodes(idx);
end


function huffman_codes = generateHuffmanCodes(node, current_code)
    if isempty(node.symbol) % Internal node
        codes_left = generateHuffmanCodes(node.left, [current_code '0']);
        codes_right = generateHuffmanCodes(node.right, [current_code '1']);
        huffman_codes = [codes_left; codes_right];
    else % Leaf node (actual symbol)
        huffman_codes = struct('symbol', node.symbol, 'code', current_code);
    end
end

function encoded_data = encodeData(data, huffman_codes)
    % Create a map of symbols to their Huffman codes
    symbol_to_code = containers.Map({huffman_codes.symbol}, {huffman_codes.code});
    
    % Encode the data
    encoded_data = [];
    for i = 1:length(data)
        encoded_data = [encoded_data symbol_to_code(data(i))];
    end
end

function rle = runLengthEncode(data)
    rle = [];
    count = 0;
    previous = data(1);
    for i = 1:length(data)
        if data(i) == previous && count < 255
            count = count + 1;
        else
            rle = [rle, previous, count];
            previous = data(i);
            count = 1;
        end
    end
    rle = [rle, previous, count]; % Add the last run
end