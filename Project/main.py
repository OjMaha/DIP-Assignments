from enc import encoder  # Ensure encoder.py is implemented and available
from dec import decoder  # Ensure decoder.py is implemented and available

import numpy as np

QM = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

def main():
    # Specify the input image
    input_image = 'images/kodak24.png'  # Provide the path to your input image here
    quality_factor = 10  # JPEG quality factor
    output_file = 'encoded_image.txt'  # File to store the encoded data

    # Perform encoding
    encoder(input_image, quality_factor, output_file, QM)

    print("Encoding done")
    print("Decoding started")

    # # Perform decoding
    decoder(output_file, quality_factor, QM)

if __name__ == "__main__":
    main()
