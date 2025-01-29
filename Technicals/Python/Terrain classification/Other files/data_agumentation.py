def image_generator(image, num_augmented_images = 5):
    # Set up the ImageDataGenerator with your desired augmentations
    datagen = ImageDataGenerator(
        rotation_range=0,             # No random rotations
        shear_range=0.2,              # Shear transformations
        zoom_range=0.2,               # Zoom range (20%)
        horizontal_flip=True,         # Horizontal flip
        vertical_flip=True,           # Vertical flip
        fill_mode='nearest'  # Use custom rotation function
    )

    img_array = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Array to hold augmented images
    augmented_images = []

    # Number of augmented images to generate
    num_augmented_images = 5

    # Generate augmented images and store them in the array
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_images.append(batch[0])  # Add the augmented image to the list
        if len(augmented_images) >= num_augmented_images:
            break  # Stop after generating the desired number of augmented images


    # Convert the list of augmented images to a NumPy array for easier processing
    augmented_images_array = np.array(augmented_images)

    aguemnted_image_array_270=np.rot90(augmented_images_array, axes=(1,2))
    aguemnted_image_array_180=np.rot90(augmented_images_array, k=2)
    aguemnted_image_array_90=np.rot90(augmented_images_array, k=3, axes=(1,2))

    # merge all the images to a single array
    all_images_array = np.concatenate((augmented_images_array, aguemnted_image_array_90, aguemnted_image_array_180, aguemnted_image_array_270))

    return all_images_array
