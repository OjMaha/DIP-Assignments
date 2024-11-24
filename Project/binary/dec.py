import numpy as np
from scipy.fftpack import idct
from matplotlib import pyplot as plt
from bitarray import bitarray
import struct

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

def decode_binary_huffman(encoded_data, huffman_codes):
    """Decodes Huffman-encoded binary data."""
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in huffman_codes:
            decoded_data.append(huffman_codes[buffer])
            buffer = ""
    if buffer:
        raise ValueError("Decoding failed: Remaining buffer contains unprocessed bits.")
    return decoded_data

def inverse_zigzag_transform(data, dims):
    """
    Reconstructs a 2D block from its zigzag-transformed 1D representation.

    Args:
        data (array-like): The 1D zigzag-transformed data.
        dims (tuple): The dimensions of the output block (e.g., (8, 8)).

    Returns:
        np.ndarray: The reconstructed 2D block.
    """
    # Define the zigzag index mapping for an 8x8 block
    idx = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])

    # Create an empty flattened array of the appropriate size
    block = np.zeros(dims[0] * dims[1])

    # Fill the block using the zigzag indices
    for i, val in enumerate(data):
        block[idx.flatten()[i]] = val

    # Reshape the block to the specified dimensions
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
    with open(encoded_file, 'rb') as f:
        # Read image dimensions and quality factor
        original_height, original_width = struct.unpack("HH", f.read(4))  # 2 bytes each
        quality_factor = struct.unpack("H", f.read(2))[0]  # 2 bytes

        # Read DC Huffman codes
        dc_huffman_codes = {}
        while True:
            # Read symbol (1 byte)
            symbol = struct.unpack("b", f.read(1))[0]
            # Break when encountering the AC Huffman codes header marker
            if symbol == -128:  # Example header end marker for DC codes
                break
            # Read code length (1 byte)
            code_len = struct.unpack("B", f.read(1))[0]
            # Read code (ceil(code_len / 8) bytes)
            code = bitarray()
            code.frombytes(f.read((code_len + 7) // 8))
            # Trim to the correct length
            code = code[:code_len].to01()  # Convert to a string of bits
            dc_huffman_codes[code] = symbol
        
        print("DC Huffman codes read")
        # print(dc_huffman_codes)

        # Read AC Huffman codes
        ac_huffman_codes = {}
        while True:
            # Read symbol (1 byte)
            symbol = struct.unpack("b", f.read(1))[0]
            if symbol == -128:  # Example header end marker for AC codes
                break
            # Read code length (1 byte)
            code_len = struct.unpack("B", f.read(1))[0]
            # Read code (ceil(code_len / 8) bytes)
            code = bitarray()
            code.frombytes(f.read((code_len + 7) // 8))
            # Trim to the correct length
            code = code[:code_len].to01()
            ac_huffman_codes[code] = symbol

        print("AC Huffman codes read")
        # print(ac_huffman_codes)

        # Read DC encoded data
        # dc_encoded_data = bitarray()
        # dc_encoded_data.fromfile(f)
        # dc_encoded_data = dc_encoded_data.to01()  # Convert to a string of bits
        dc_encoded_data = bitarray()
        # Read until -128 occurs
        while True:
            byte = f.read(1)
            if not byte:
                break
            if struct.unpack("b", byte)[0] == -128:
                break
            dc_encoded_data.frombytes(byte)

        print("DC",len(dc_encoded_data))

        # Read AC encoded data
        ac_encoded_data = bitarray()
        ac_encoded_data.fromfile(f)
        ac_encoded_data = ac_encoded_data.to01()
        print("AC",len(ac_encoded_data))

    # Decode DC coefficients
    decoded_dc_data = decode_binary_huffman(dc_encoded_data, dc_huffman_codes)

    # Decode AC coefficients
    decoded_ac_data = decode_binary_huffman(ac_encoded_data, ac_huffman_codes)

    # Reconstruct DC coefficients
    DC_differences = decoded_dc_data
    DC_coefficients = [DC_differences[0]]  # The first DC value is absolute
    for diff in DC_differences[1:]:
        DC_coefficients.append(DC_coefficients[-1] + diff)

    # Modify quantization matrix according to quality factor
    Q = Q * (50 / quality_factor)
    Q[Q == 0] = 1  # Avoid division by zero

    # Calculate padded dimensions
    pad_height = (8 - original_height % 8) % 8
    pad_width = (8 - original_width % 8) % 8
    padded_height = original_height + pad_height
    padded_width = original_width + pad_width
    compressed_image = np.zeros((padded_height, padded_width))

    # Reconstruct the image
    patch_number = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            # Get DC coefficient for this block
            dc_value = DC_coefficients[patch_number]

            # Collect AC coefficients for the current block
            ac_coefficients = decoded_ac_data[patch_number * 63:(patch_number + 1) * 63]

            # Combine DC and AC coefficients
            zigzag_coeffs = [dc_value] + ac_coefficients

            # Dequantize and inverse transform
            quantized_block = inverse_zigzag_transform(zigzag_coeffs, (8, 8))
            compressed_image[i:i + 8, j:j + 8] = quantized_block
            patch_number += 1

    # Perform dequantization and inverse DCT
    decompressed_image = np.zeros_like(compressed_image)
    for i in range(0, compressed_image.shape[0], 8):
        for j in range(0, compressed_image.shape[1], 8):
            quantized_block = compressed_image[i:i + 8, j:j + 8]
            dequantized_block = quantized_block * Q
            decompressed_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
            decompressed_image[i:i + 8, j:j + 8] = decompressed_block

    # Shift pixel values back by adding 128
    decompressed_image = decompressed_image + 128

    # Clamp values to the [0, 255] range and crop to original dimensions
    decompressed_image = np.clip(decompressed_image, 0, 255).astype(np.uint8)
    decompressed_image = decompressed_image[:original_height, :original_width]

    return decompressed_image

