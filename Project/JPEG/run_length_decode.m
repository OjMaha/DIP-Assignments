function [ac_coeffs, idx] = run_length_decode(rle, idx)
    % Run-Length Decoding for AC coefficients
    ac_coeffs = [];
    for i = idx:size(rle, 1) % Use built-in size function here
        run_length = rle(i, 1);
        coeff_size = rle(i, 2); % Rename 'size' to 'coeff_size'
        value = rle(i, 3);
        
        if run_length == 0 && coeff_size == 0 && value == 0
            % End-of-block (EOB)
            break;
        elseif run_length == 15 && coeff_size == 0 && value == 0
            % Special case for 16 zeros
            ac_coeffs = [ac_coeffs, zeros(1, 16)];
        else
            % Add zeros followed by the actual value
            ac_coeffs = [ac_coeffs, zeros(1, run_length), value];
        end
    end
    idx = idx + length(ac_coeffs);
end
