function [dc_encoded, huff_dict] = huffman_encode_dc(dc_diff)
    % Huffman encoding for DC coefficients

    % Find unique symbols
    symbols = unique(dc_diff);

    % Handle edge case: Only one unique symbol
    if isscalar(symbols)
        huff_dict = {symbols, {'0'}}; % Assign code '0' to the only symbol
        dc_encoded = repmat({'0'}, size(dc_diff)); % Encode all as '0'
        save('huffman_dict.mat', 'huff_dict'); % Save dictionary
        return;
    end

    % Calculate probabilities of each symbol
    probs = histcounts(dc_diff, [symbols - 0.5, max(symbols) + 0.5]) / numel(dc_diff);

    % Generate Huffman dictionary
    huff_dict = huffmandict(symbols, probs);

    % Save the dictionary for decoding
    save('huffman_dict.mat', 'huff_dict');

    % Encode using the dictionary
    dc_encoded = huffmanenco(dc_diff, huff_dict);
end
