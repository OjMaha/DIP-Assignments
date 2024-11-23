import numpy as np
from scipy.fftpack import idct
from matplotlib import pyplot as plt

def decode_huffman(encoded_data, huffman_codes):
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in huffman_codes:
            decoded_data.append(huffman_codes[buffer])
            buffer = ""
    if buffer:
        print("Buffer: ", buffer)
        raise ValueError("Decoding failed: Remaining buffer contains unprocessed bits.")
    return decoded_data

def inverse_zigzag_transform(data, dims):
    idx = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]
    block = np.zeros(64)
    for i, val in enumerate(data):
        block[idx[i]] = val
    return block.reshape(dims)

def run_length_decode(encoded_data):
    decoded_data = []
    block = []
    for code, zero_count in encoded_data:
        if code == '-1' and zero_count == -1:
            while len(block) < 63:
                block.append(0)
            decoded_data.append(block)
            block = []
        else:
            # Append the zeros followed by the decoded coefficient
            block.extend([0] * zero_count)  # Add zeros
            block.append(int(code))  # Add the non-zero coefficient

    return decoded_data

def decoder(encoded_file, quality_factor, Q):
    with open(encoded_file, 'r') as f:
        # Parse image dimensions
        dimensions_line = f.readline().strip()
        dimensions = dimensions_line.split(':')[1].strip()  # Extract "512x768"
        original_height, original_width = map(int, dimensions.split('x'))

        # Parse quality factor
        quality_line = f.readline().strip()
        parsed_quality = int(quality_line.split(':')[1].strip())
        assert parsed_quality == quality_factor, "Quality factor mismatch."
        
        found_AC = False
        found_DC_encode = False
        
        # Parse DC Huffman codes
        dc_huffman_header = f.readline().strip()
        if dc_huffman_header != "DC Huffman Codes:":
            raise ValueError("Unexpected format: Expected 'DC Huffman Codes:' header")

        dc_huffman_codes = {}
        while True:
            line = f.readline().strip()
            if line == "AC Huffman Codes:":
                found_AC = True
                break  # Move to AC Huffman codes section
            if ':' not in line:
                raise ValueError("Unexpected format in DC Huffman Codes section.")
            symbol, code = line.split(':', 1)  # Allow for ':' in the code
            dc_huffman_codes[code.strip()] = int(symbol.strip())  
            
        if not found_AC:
            ac_huffman_header = f.readline().strip()
        else:
            ac_huffman_header = "AC Huffman Codes:"
            
        if ac_huffman_header != "AC Huffman Codes:":
            raise ValueError("Unexpected format: Expected 'AC Huffman Codes:' header")

        ac_huffman_codes = {}
        while True:
            line = f.readline().strip()
            if line.startswith("Encoded DC Data:"):
                found_DC_encode = True
                break  # Move to Encoded Data section
            if ':' not in line:
                raise ValueError("Unexpected format in AC Huffman Codes section.")
            symbol, code = line.split(':', 1)  # Allow for ':' in the code
            ac_huffman_codes[code.strip()] = symbol.strip()  

        # Read DC encoded data
        if found_DC_encode:
            dc_encoded_data_header = "Encoded DC Data:"
        else:
            dc_encoded_data_header = f.readline().strip()
        if dc_encoded_data_header != "Encoded DC Data:":
            raise ValueError("Unexpected format: Expected 'Encoded DC Data:' header")
        
        dc_encoded_data = f.readline().strip()  # Extract actual DC encoded data
        # print("got DC encoded data: ", dc_encoded_data)
        
        # Read AC encoded data
        ac_encoded_data_header = f.readline().strip()
        if ac_encoded_data_header != "Encoded AC Data:":
            raise ValueError("Unexpected format: Expected 'Encoded AC Data:' header")
        
        ac_encoded_data = []
        while True:
            line = f.readline().strip()
            if line == "":
                break
            if ':' not in line:
                print("line: ", line)
                raise ValueError("Unexpected format in AC Huffman Codes section.")
            code, count_zero = line.split(':', 1)  # Allow for ':' in the code
            ac_encoded_data.append((code.strip(), int(count_zero.strip())))             
        
    # Debug prints for Huffman tables
    # print("DC Huffman Codes:")
    # for symbol, code in dc_huffman_codes.items():
    #     print(f"{symbol}: {code}")

    # print("AC Huffman Codes:")
    # for symbol, code in ac_huffman_codes.items():
    #     print(f"{symbol}: {code}")
        
    # Decode Huffman data for DC and AC coefficients
    decoded_dc_data = decode_huffman(dc_encoded_data, dc_huffman_codes)
    print("Decoding DC Huffman done")
    
    AC_rle = []
    for (code, count_zero) in ac_encoded_data:
        if code == "-1" and count_zero == -1:
            AC_rle.append((code, count_zero))
        else:
            AC_rle.append((ac_huffman_codes[code], count_zero))
            
    print("Decoding AC Huffman done")
    # print("AC RLE: ", AC_rle)
    
    # Reconstruct AC coefficients
    #it is an array of arrays of length 63 representing the AC coefficients of each block
    decoded_ac_data = run_length_decode(AC_rle)
    
    # Reconstruct DC coefficients
    DC_differences = decoded_dc_data

    DC_coefficients = [DC_differences[0]]  # The first DC value is absolute
    for diff in DC_differences[1:]:
        DC_coefficients.append(DC_coefficients[-1] + diff)

    #modify quantization matrix according to quality factor
    Q = Q * (50 / quality_factor)
    Q[Q == 0] = 1  # Avoid division by zero

    # Calculate padded dimensions
    pad_height = (8 - original_height % 8) % 8
    pad_width = (8 - original_width % 8) % 8
    padded_height = original_height + pad_height
    padded_width = original_width + pad_width
    compressed_image = np.zeros((padded_height, padded_width))

    assert(len(decoded_ac_data) == len(DC_coefficients))

    patch_number = 0   
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            # Get DC coefficient for this block
            dc_value = DC_coefficients[patch_number]

            zigzag_coeffs = [dc_value]  # Initialize with DC value
            zigzag_coeffs.extend(decoded_ac_data[patch_number])

            quantized_block = inverse_zigzag_transform(zigzag_coeffs, (8, 8))
            compressed_image[i:i+8, j:j+8] = quantized_block
            patch_number += 1

    print("Patches regenerated")

    # Perform dequantization and inverse DCT
    decompressed_image = np.zeros_like(compressed_image)
    for i in range(0, compressed_image.shape[0], 8):
        for j in range(0, compressed_image.shape[1], 8):
            quantized_block = compressed_image[i:i+8, j:j+8]
            dequantized_block = quantized_block * Q
            decompressed_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
            decompressed_image[i:i+8, j:j+8] = decompressed_block

    # Shift pixel values back by adding 128
    decompressed_image = decompressed_image + 128

    # Clamp values to the [0, 255] range and crop to original dimensions
    decompressed_image = np.clip(decompressed_image, 0, 255).astype(np.uint8)
    decompressed_image = decompressed_image[:original_height, :original_width]

    print("Decompression done")

    # Display the reconstructed image
    plt.imshow(decompressed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.show()

