import numpy as np
from scipy.fftpack import idct
from matplotlib import pyplot as plt

def decoder(encoded_file, quality_factor):
    with open(encoded_file, 'r') as f:
        # Parse image dimensions
        dimensions_line = f.readline().strip()
        dimensions = dimensions_line.split(':')[1].strip()  # Extract "512x768"
        original_height, original_width = map(int, dimensions.split('x'))

        # Parse quality factor
        quality_line = f.readline().strip()
        parsed_quality = int(quality_line.split(':')[1].strip())
        assert parsed_quality == quality_factor, "Quality factor mismatch."

        # Skip the descriptive header for Huffman Codes
        huffman_header = f.readline().strip()
        if huffman_header != "Huffman Codes:":
            raise ValueError("Unexpected format: Expected 'Huffman Codes:' header")

        # Read Huffman codes
        huffman_codes = {}
        while True:
            line = f.readline().strip()
            if line.startswith("Encoded Data:"):
                break  # Stop at the encoded data section
            symbol, code = line.split(':')
            huffman_codes[code.strip()] = float(symbol.strip())

        # Read the encoded data (strip "Encoded Data:" label)
        encoded_data = line.split(':', 1)[1].strip()

    # Decode the Huffman-encoded data
    decoded_data = decode_huffman(encoded_data, huffman_codes)

    print("Decoding Huffman done")
    #print huffman table
    print("Huffman Codes:")
    for symbol, code in huffman_codes.items():
        print(f"{symbol}: {code}")

    # Define the JPEG quantization matrix
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    Q = Q * (50 / quality_factor)
    Q[Q == 0] = 1  # Avoid division by zero

    # Calculate padded dimensions
    pad_height = (8 - original_height % 8) % 8
    pad_width = (8 - original_width % 8) % 8
    padded_height = original_height + pad_height
    padded_width = original_width + pad_width
    compressed_image = np.zeros((padded_height, padded_width))

    # Reconstruct compressed_image from decoded RLE data
    rle_index = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            zigzag_coeffs, rle_index = run_length_decode(decoded_data, rle_index)
            quantized_block = inverse_zigzag_transform(zigzag_coeffs, (8, 8))
            compressed_image[i:i+8, j:j+8] = quantized_block

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

    # Display the reconstructed image
    plt.imshow(decompressed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.show()

def decode_huffman(encoded_data, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += bit
        if buffer in reverse_codes:
            decoded_data.append(reverse_codes[buffer])
            buffer = ""
    if buffer:
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

def run_length_decode(encoded_data, start_idx):
    decoded = []
    idx = start_idx
    while len(decoded) < 64 and idx < len(encoded_data) - 1:
        symbol = encoded_data[idx]
        count = encoded_data[idx + 1]
        to_add = min(count, 64 - len(decoded))
        decoded.extend([symbol] * to_add)
        idx += 2
    decoded = decoded[:64]  # Ensure exactly 64 coefficients
    return decoded, idx
