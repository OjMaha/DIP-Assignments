from enc import encoder  
from dec import decoder  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    quality_factor = 50  # JPEG quality factor
    output_file = 'encoded_image.txt'  # File to store the encoded data

    # Perform encoding
    encoder(input_image, quality_factor, output_file, QM)

    print("Encoding done")
    print("Decoding started")


    reconstructed_img = decoder(output_file, quality_factor, QM)

    print("Decoding done")
    
    # Save the reconstructed image on grayscale
    #filename should have input_image name with quality factor
    reconstructed_img = Image.fromarray(reconstructed_img)
    reconstructed_img.save(f'{input_image.split('/')[1].split('.')[0]}_{quality_factor}.png')

if __name__ == "__main__":
    main()
