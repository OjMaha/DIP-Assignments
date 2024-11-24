import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from PIL import Image
import math
from heapq import heappush, heappop
from collections import defaultdict

# Helper Functions
def zigzag_transform(block):
    idx = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]).flatten()
    return block.flatten()[idx]

def run_length_encode(data, huffman_table):
    encoded_data = []
    zero_count = 0

    for coeff in data:
        if coeff == 0:
            zero_count += 1
            # if zero_count == 16:  
            #     encoded_data.append((15, 0))
            #     zero_count = 0
        else:
            huffman_code = huffman_table[coeff]
            encoded_data.append((huffman_code, zero_count))
            zero_count = 0

    encoded_data.append((-1,-1))
        
    return encoded_data

def huffman_encode(symbols, frequencies, exclude_zeros=False):
    # Step 1: Filter out zeros if exclude_zeros is True
    if exclude_zeros:
        filtered_symbols = []
        filtered_frequencies = []
        for sym, freq in zip(symbols, frequencies):
            if sym != 0:
                filtered_symbols.append(sym)
                filtered_frequencies.append(freq)
        symbols, frequencies = filtered_symbols, filtered_frequencies

    # Step 2: Create a priority queue (min-heap) with [frequency, symbol/tree]
    heap = [[freq, [sym, ""]] for sym, freq in zip(symbols, frequencies)]
    # Heapify the list for proper min-heap behavior
    heap.sort()  # Sort ensures ascending order of frequency

    # Step 3: Build the Huffman tree
    while len(heap) > 1:
        # Extract the two nodes with the lowest frequencies
        lo = heappop(heap)
        hi = heappop(heap)

        # Combine nodes to create a parent node
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]  # Add '0' to the code for left child
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]  # Add '1' to the code for right child
        new_node = [lo[0] + hi[0]] + lo[1:] + hi[1:]  # Combine frequencies and children

        # Insert the new parent node back into the heap
        heappush(heap, new_node)
        heap.sort()  # Ensure the heap remains sorted

    # Step 4: Extract Huffman codes from the final tree
    huffman_tree = heappop(heap)  # Root node of the Huffman tree
    huffman_codes = {symbol: code for symbol, code in huffman_tree[1:]}  # Collect codes

    # Sort by code length for a more human-readable output
    huffman_codes = dict(sorted(huffman_codes.items(), key=lambda item: (len(item[1]), item[0])))

    return huffman_codes

def rgb_to_ycbcr(R,G,B):
    """Convert RGB to YCbCr."""
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr

def downsample(channel):
    """Downsample by 2 in both dimensions."""
    return channel[::2, ::2]

def encode_image(image_path, quality_factor, file_Y, file_Cb, file_Cr, Q):
    """Encode an RGB image."""
    image = Image.open(image_path).convert('RGB')
    
    # Convert image to a NumPy array for pixel-wise operations
    rgb_array = np.array(image)

    # Extract the R, G, B channels
    R = rgb_array[:, :, 0]
    G = rgb_array[:, :, 1]
    B = rgb_array[:, :, 2]
    
    print(R)
    print(G)
    print(B)
    
    # Convert to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(R,G,B)
    
    # Convert to uint8 (integer pixel values)
    Y = Y.astype(np.uint8)
    Cb = Cb.astype(np.uint8)
    Cr = Cr.astype(np.uint8)
    
    # Downsample Cb and Cr
    Cb_downsampled = downsample(Cb)
    Cr_downsampled = downsample(Cr)
    
    # print(Y)
    # print(Cb_downsampled)
    # print(Cr_downsampled)

    # Compress each channel
    compressed_Y = encoder(Y, quality_factor, file_Y, Q)
    compressed_Cb = encoder(Cb_downsampled, quality_factor, file_Cb, Q)
    compressed_Cr = encoder(Cr_downsampled, quality_factor, file_Cr, Q)

    return compressed_Y, compressed_Cb, compressed_Cr

def encoder(image, quality_factor, output_file, Q):
    image = image.astype(np.float64)
    # Store original dimensions
    original_height, original_width = image.shape

    # Ensure the image dimensions are multiples of 8
    pad_height = (8 - original_height % 8) % 8
    pad_width = (8 - original_width % 8) % 8
    image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    height, width = image.shape

    # Define JPEG quantization matrix for grayscale (luminance)
    Q = Q * (50 / quality_factor)
    Q[Q == 0] = 1  # Avoid division by zero

    # Shift the image values by -128 and prepare for compression
    sht_image = image - 128
    compressed_image = np.zeros_like(sht_image)
    
     # Initialize storage for DC and AC coefficients
    DC_differences = []
    AC_coefficients = []
    
    # Compress the image in 8x8 blocks with DCT and quantization
    prev_dc = 0

    # Compress the image in 8x8 blocks with DCT and quantization
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = sht_image[i:i+8, j:j+8]
            
            # Apply DCT using scipy (2D DCT)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Quantize the DCT block
            quantized_block = np.round(dct_block / Q)
            compressed_image[i:i+8, j:j+8] = quantized_block
            
            # Extract DC and AC coefficients
            zigzag_coeffs = zigzag_transform(quantized_block)
            dc_value = zigzag_coeffs[0]
            ac_values = zigzag_coeffs[1:]

            # Compute DC difference and store it
            dc_diff = dc_value - prev_dc
            DC_differences.append(dc_diff)
            prev_dc = dc_value

            # Store AC coefficients
            AC_coefficients.extend(ac_values)
    
    #make AC_coefficients an integer array
    AC_coefficients = np.array(AC_coefficients, dtype=int)

    #make DC_differences an integer array
    DC_differences = np.array(DC_differences, dtype=int)

    # Huffman encoding for DC differences
    unique_dc, counts_dc = np.unique(DC_differences, return_counts=True)
    dc_huffman_codes = huffman_encode(unique_dc, counts_dc)

    # Huffman encoding for AC coefficients
    unique_ac, counts_ac = np.unique(AC_coefficients, return_counts=True)
    ac_huffman_codes = huffman_encode(unique_ac, counts_ac, True)

    # # Debug print for DC Huffman codes
    # print("DC Huffman Codes:")
    # for symbol, code in dc_huffman_codes.items():
    #     print(f"{symbol}: {code}")

    # print("AC Huffman Codes:")
    # for symbol, code in ac_huffman_codes.items():
    #     print(f"{symbol}: {code}")
        
    # Perform Huffman encoding for DC coefficients
    dc_encoded_data = []
    for diff in DC_differences:
        huffman_code = dc_huffman_codes[diff]
        dc_encoded_data.append(huffman_code)
        
    # Perform RLE encoding for AC coefficients for each block
    ac_encoded_data = []
    for i in range(0, len(AC_coefficients), 63):
        block_data = AC_coefficients[i:i+63]
        encoded_block = run_length_encode(block_data, ac_huffman_codes)
        ac_encoded_data.extend(encoded_block) 

    # Save metadata and encoded data to a .txt file
    with open(output_file, 'w') as f:
        # Write metadata
        f.write(f"Original Dimensions: {original_height}x{original_width}\n")
        f.write(f"Quality Factor: {quality_factor}\n")
        
        # Write Huffman codes for DC
        f.write("DC Huffman Codes:\n")
        for symbol, code in dc_huffman_codes.items():
            f.write(f"{symbol}: {code}\n")
        
        # Write Huffman codes for AC
        f.write("AC Huffman Codes:\n")
        for symbol, code in ac_huffman_codes.items():
            f.write(f"{symbol}: {code}\n")
            
        # Write encoded DC data
        f.write("Encoded DC Data:\n")
        f.write("".join(dc_encoded_data))
        f.write("\n")
        
        # Write encoded AC data
        f.write("Encoded AC Data:\n")
        for symbol, count in ac_encoded_data:
            f.write(f"{symbol}:{count}\n")

    print(f"Encoded data and metadata saved to {output_file}")
