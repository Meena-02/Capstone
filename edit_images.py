import os
from PIL import Image
from rembg import remove

# Function to create a folder if it does not exist
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

# Function to remove background, resize, and save the image
def process_image(image_path, output_folder):
    # Open the image
    image = Image.open(image_path)
    
    # Remove the background
    image_no_bg = remove(image)
    
    # Resize the image to 500x500 pixels
    resized_image = image_no_bg.resize((500, 500), Image.Resampling.LANCZOS)
    
    # Save the image as PNG in the new folder
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".png")
    resized_image.save(output_path, "PNG")
    print(f"Processed and saved: {output_path}")

# Main function
def main(input_folder, output_folder):
    # Create the output folder if it does not exist
    create_folder(output_folder)
    
    # Read images from the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        
        # Check if the file is an image
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            process_image(image_path, output_folder)
        else:
            print(f"Skipping non-image file: {image_name}")

if __name__ == "__main__":
    # Define the input and output folders
    input_folder = "Dataset/Visual/VP005/ref_images"  # Replace with your input folder path
    output_folder = "output_images"  # Replace with your output folder path
    
    # Run the main function
    main(input_folder, output_folder)