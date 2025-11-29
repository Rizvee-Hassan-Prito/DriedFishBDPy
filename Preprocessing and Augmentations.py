


#### Preprocessing Script ####
import os
import cv2

def resize_image_opencv(input_path, output_path, size=(1024, 1024)):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Could not read: {input_path}")
            return
        
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as JPEG
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_dataset(input_folder, output_folder, size=(1024, 1024)):
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):

                input_path = os.path.join(root, file)

                # Recreate folder structure inside output folder
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # Always save as JPEG
                output_filename = os.path.splitext(file)[0] + ".jpg"
                output_path = os.path.join(output_dir, output_filename)

                resize_image_opencv(input_path, output_path, size)


if __name__ == "__main__":
    input_folder = r"C:/Path/To/Original_Dataset"
    output_folder = r"C:/Path/To/Resized_Dataset"

    process_dataset(input_folder, output_folder, size=(1024, 1024))


#%%

#### Augmentation Script ####

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

folder_path= r"C:/Path/To/Original_Dataset"
aug_folder_path = r"C:/Path/To/Resized_Dataset"
fl=os.listdir(folder_path)

for i in fl:
    
     fl_sub= os.listdir(folder_path+str(i))
     for j in fl_sub:

        input_folder=folder_path+str(i)+"/" +str(j)+"/"
        output_folder = aug_folder_path+str(i)+"/" +str(j)+"/"
        

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Augmentation methods
        def flip_horizontal(image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
    
        def flip_vertical(image):
            return image.transpose(Image.FLIP_TOP_BOTTOM)
    
        def rotate_90_clockwise(image):
            return image.rotate(-90, expand=False)
    
        def rotate_90_counterclockwise(image):
            return image.rotate(90, expand=False)
    
        def rotate_180(image):
            return image.rotate(180, expand=False)
    
        def rotate(image, angle):
            return image.rotate(angle, expand=False)
    
        def adjust_brightness(image, factor):
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
    
        def apply_blur(image, radius):
            return image.filter(ImageFilter.GaussianBlur(radius))
    
        def add_noise(image, noise_factor):
            np_image = np.array(image)
            noise_pixels = int(noise_factor * np_image.size)
            for _ in range(noise_pixels):
                x = random.randint(0, np_image.shape[0] - 1)
                y = random.randint(0, np_image.shape[1] - 1)
                np_image[x, y] = random.randint(0, 255)  
            return Image.fromarray(np_image)
    
        augmentations = [
            flip_vertical,
            flip_horizontal,
            rotate_90_clockwise,
            rotate_90_counterclockwise,
            rotate_180,
            lambda img: rotate(img, -25),
            lambda img: rotate(img, 25),
            lambda img: adjust_brightness(img, 0.7), 
            lambda img: adjust_brightness(img, 1.3),
            lambda img: apply_blur(img, 2.5),  
            lambda img: add_noise(img, 0.02)
        ]

        for filename in os.listdir(input_folder):
            
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                
                image_path = os.path.join(input_folder, filename)
                
                image = Image.open(image_path)
                
                image.save(os.path.join(output_folder, filename))
                

                aug_image = image.copy()
                x=1
                selected_augmentations = augmentations
                for aug in selected_augmentations:
                    aug_pic = aug(aug_image)
                    aug_pic.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{x}.JPG"))
                    x+=1
                print(f"Augmentating {i}!")


