function rle = run_length_encode(ac_coeffs)
    % Run-Length Encoding for AC coefficients
    rle = [];
    zero_count = 0;
    
    for i = 1:length(ac_coeffs)
        if ac_coeffs(i) == 0
            zero_count = zero_count + 1;
            if zero_count == 16 % Special case for 16 zeros
                rle = [rle; 15, 0, 0]; % (15, 0, 0) indicates 16 zeros
                zero_count = 0;
            end
        else
            % Normal case: (run-length, size, value)
            rle = [rle; zero_count, length(dec2bin(ac_coeffs(i))), ac_coeffs(i)];
            zero_count = 0;
        end
    end
    
    % End-of-block (EOB) marker
    rle = [rle; 0, 0, 0];
end

function [ac_coeffs, idx] = run_length_decode(rle, idx)
    % Run-Length Decoding for AC coefficients
    ac_coeffs = [];
    for i = idx:size(rle, 1)
        run_length = rle(i, 1);
        size = rle(i, 2);
        value = rle(i, 3);
        
        if run_length == 0 && size == 0 && value == 0
            % End-of-block (EOB)
            break;
        elseif run_length == 15 && size == 0 && value == 0
            % Special case for 16 zeros
            ac_coeffs = [ac_coeffs, zeros(1, 16)];
        else
            % Add zeros followed by the actual value
            ac_coeffs = [ac_coeffs, zeros(1, run_length), value];
        end
    end
    idx = idx + length(ac_coeffs);
end
