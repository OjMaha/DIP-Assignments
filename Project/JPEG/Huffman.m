function [dc_encoded, huff_dict] = huffman_encode_dc(dc_diff)
    % Huffman encoding for DC coefficients
    % Generate Huffman dictionary dynamically
    symbols = unique(dc_diff); % All possible symbols
    probs = histcounts(dc_diff, [symbols-0.5, max(symbols)+0.5]) / numel(dc_diff);
    huff_dict = huffmandict(symbols, probs);
    
    % Encode using the dictionary
    dc_encoded = huffmanenco(dc_diff, huff_dict);
end

function [dc_diff, idx] = huffman_decode_dc(encoded_data, idx)
    % Huffman decoding for DC coefficients
    % Note: Requires the same dictionary used for encoding
    load('huffman_dict.mat', 'huff_dict'); % Load pre-saved dictionary
    dc_diff = huffmandeco(encoded_data(idx:end), huff_dict);
    idx = idx + numel(dc_diff);
end
