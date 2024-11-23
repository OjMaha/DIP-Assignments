from enc import encoder  # Ensure encoder.py is implemented and available
from dec import decoder  # Ensure decoder.py is implemented and available

def main():
    # Specify the input image
    input_image = 'images/kodak24.png'  # Provide the path to your input image here
    quality_factor = 50  # JPEG quality factor
    output_file = 'encoded_image.txt'  # File to store the encoded data

    # Perform encoding
    encoder(input_image, quality_factor, output_file)

    print("Encoding done")
    # print("Decoding started")

    # # Perform decoding
    # decoder(output_file, quality_factor)

if __name__ == "__main__":
    main()
