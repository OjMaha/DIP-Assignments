function [dc_diff, idx] = huffman_decode_dc(encoded_data, idx)
    % Huffman decoding for DC coefficients
    % Note: Requires the same dictionary used for encoding
    
    % Load the Huffman dictionary
    load('huffman_dict.mat', 'huff_dict'); % Load pre-saved dictionary
    
    % Check if the dictionary contains only one symbol
    if isscalar(huff_dict{1})
        % Special case: Single unique symbol in the dictionary
        single_symbol = huff_dict{1}(1); % The only symbol in the dictionary
        num_symbols = length(encoded_data(idx:end)); % Count remaining encoded data
        dc_diff = repmat(single_symbol, 1, num_symbols); % Decode all as the single symbol
        idx = idx + num_symbols; % Update index
    else
        % General case: Multiple symbols in the dictionary
        dc_diff = huffmandeco(encoded_data(idx:end), huff_dict);
        idx = idx + numel(dc_diff); % Update index
    end
end
