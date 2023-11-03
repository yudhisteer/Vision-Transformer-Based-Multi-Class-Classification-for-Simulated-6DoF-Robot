import os
import shutil
from typing import List
import shutil
import random
from helper_functions import walk_through_dir


# Function to retrieve images from solo folder
def extract_img (source_folder: str, target_folder: str, skip_first=True, name_file="class") -> None:
    index = 1
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.endswith(".png"):
                source_file = os.path.join(root, filename)

                if skip_first:
                    skip_first = False
                    continue  # Skip the first file because first image is blank (plane only)

                new_filename = f"{name_file}_{index}.jpeg"
                target_file = os.path.join(target_folder, new_filename)
                shutil.copy2(source_file, target_file)
                index += 1
                print(f"Saved {target_file}")





# Function to create train and test and sub-folders for classes
def create_train_test_folders(main_folder_path: str, classes: List[str]) -> None:

    """
    :param main_folder_path: main folder where test and train folder will be created,
    :param classes: Classes name for image classification task: e.g, classes=['box', 'plate', 'vase']
    :return: None
    """

    # check if main folder exist
    if not os.path.exists(main_folder_path):
        raise ValueError(f"'main_folder_path' is not a valid path: {main_folder_path}")

    # Create train and test folders
    test_folder_path = os.path.join(main_folder_path, 'test')
    train_folder_path = os.path.join(main_folder_path, 'train')
    os.makedirs(main_folder_path, exist_ok=True)
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    # Create sub-folders in test and train folder
    for class_ in classes:
        class_folder_path_test = os.path.join(main_folder_path, 'test', class_)
        class_folder_path_train = os.path.join(main_folder_path, 'train', class_)
        os.makedirs(class_folder_path_test, exist_ok=True)
        os.makedirs(class_folder_path_train, exist_ok=True)



def split_images(source_folder, train_folder, test_folder, split_ratio=0.8):
    if not os.path.exists(source_folder):
        raise ValueError("Source folder does not exist.'")

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # List all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    num_images = len(image_files)
    print("Number of images: ", num_images)

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the split point
    split_index = int(split_ratio * num_images)

    # Copy images to train and test folders
    for i, image in enumerate(image_files):
        source_path = os.path.join(source_folder, image)
        if i < split_index:
            destination_path = os.path.join(train_folder, image)
        else:
            destination_path = os.path.join(test_folder, image)
        shutil.copy(source_path, destination_path)

    print(f"Split {split_ratio*num_images} of images to train folder and {((1-split_ratio)*num_images)} to test folder.")







if __name__ == "__main__":

    # # check directory
    # walk_through_dir(cardboard_folder_img_path)

    # Get the current directory
    current_directory = os.getcwd()

    # parent directory
    parent_directory = os.path.dirname(current_directory)

    # Source folders
    cardboard_folder_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Cardboard')
    plate_folder_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Plate')
    vase_folder_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Vase')

    # Target folders
    cardboard_folder_img_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Cardboard_Img')
    plate_folder_img_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Plate_Img')
    vase_folder_img_path = os.path.join(parent_directory, 'Data', 'Generated_Data', 'Vase_Img')
    os.makedirs(cardboard_folder_img_path, exist_ok=True)
    os.makedirs(plate_folder_img_path, exist_ok=True)
    os.makedirs(vase_folder_img_path, exist_ok=True)


    # Set a flag to control whether the function should run (Extracting image from solo folder)
    extract_img_function = False
    if extract_img_function:
        print("*****---------Extracting Images-----------***")
        # Extract cardboard images from solo folder
        extract_img(source_folder=cardboard_folder_path, target_folder=cardboard_folder_img_path, name_file="cardboard")
        # Extract plate images from solo folder
        extract_img(source_folder=plate_folder_path, target_folder=plate_folder_img_path, name_file="plate")
        # Extract vase images from solo folder
        extract_img(source_folder=vase_folder_path, target_folder=vase_folder_img_path, name_file="vase")


    # Generate main folder
    main_folder_name = "cardboard_plate_vase"
    main_folder_path = os.path.join(parent_directory, 'Data', main_folder_name)
    print(main_folder_path)
    os.makedirs(main_folder_path, exist_ok=True)


    # Flag: Creating test and train folders + sub-folders
    create_train_test_folders_function = False
    if create_train_test_folders_function:
        print("*****---------Creating Test and Train Folders-----------***")
        # Create test and train sub-folders
        create_train_test_folders(main_folder_path, classes=['cardboard', 'plate', 'vase'])
        print("Creating Test and Train Folders successful!")

    # Flag: Train-Test split
    split_images_function = False
    if split_images_function:
        classes = ['cardboard', 'plate', 'vase']
        folders_path = [cardboard_folder_img_path, plate_folder_img_path, vase_folder_img_path]
        for class_, folder_path in zip(classes, folders_path):
            train_folder_class_path = os.path.join(parent_directory, 'Data', main_folder_name, 'train', class_)
            test_folder_class_path = os.path.join(parent_directory, 'Data', main_folder_name, 'test', class_)
            split_images(folder_path, train_folder_class_path, test_folder_class_path, split_ratio=0.8)















