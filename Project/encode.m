encoder('kodak24.png', 50, 'compressed_data');

function encoder(image_path, quality_factor, output_file)
    % Load the grayscale image
    image = imread(image_path);
    if size(image, 3) == 3
        image = rgb2gray(image); % Convert to grayscale if it's an RGB image
    end
    image = double(image); % Convert to double for processing

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

    % Shift the image values by -128
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

    % Huffman encoding
    [unique_symbols, ~, idx] = unique(compressed_image(:));
    frequencies = accumarray(idx, 1);
    [~, huffman_codes] = huffman_manual(unique_symbols, frequencies);
    encoded_data = encodeData(compressed_image(:), huffman_codes);

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
        fwrite(fid, huffman_codes{i}.symbol, 'int32'); % Write symbol
        fwrite(fid, length(huffman_codes{i}.code), 'uint32'); % Write code length
        fwrite(fid, huffman_codes{i}.code - '0', 'ubit1'); % Write code as binary
    end

    % Write encoded data length and content
    fwrite(fid, length(encoded_data), 'uint32');
    fwrite(fid, encoded_data - '0', 'ubit1');

    fclose(fid);
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

% Helper function to generate Huffman codes from the tree
function huffman_codes = generateHuffmanCodes(node, current_code)
    if isempty(node.symbol) % Internal node
        huffman_codes = [];
        if ~isempty(node.left)
            huffman_codes = [huffman_codes; generateHuffmanCodes(node.left, [current_code '0'])];
        end
        if ~isempty(node.right)
            huffman_codes = [huffman_codes; generateHuffmanCodes(node.right, [current_code '1'])];
        end
    else % Leaf node (actual symbol)
        huffman_codes = {struct('symbol', node.symbol, 'code', current_code)};
    end
end

% Helper function to encode the data based on the Huffman codes
function encoded_data = encodeData(data, huffman_codes)
    % Map each symbol to its corresponding Huffman code
    symbol_to_code = containers.Map();
    for i = 1:length(huffman_codes)
        symbol_to_code(num2str(huffman_codes{i}.symbol)) = huffman_codes{i}.code;
    end
    
    % Encode the data using the symbol-to-code map
    encoded_data = '';
    for i = 1:length(data)
        encoded_data = [encoded_data symbol_to_code(num2str(data(i)))];
    end
end
