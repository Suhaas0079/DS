agumented_images = np.array(np.copy(data))
agumented_label_list = np.copy(labels).flatten()

for i in range(len(data)):
    # Extract the image and its corresponding label
    images_result= image_generator(data[i, :, :, :])
    # Append the augmented images to the list
    agumented_images = np.concatenate((agumented_images, images_result))

    repeated_labels = np.repeat(labels[i], images_result.shape[0])  # Repeat the label for the batch of images
    
    # Append the repeated labels
    agumented_label_list = np.concatenate((agumented_label_list, repeated_labels))

split_list_cnn = train_test_split(agumented_images,
                              agumented_label_list,
                              test_size = 0.1,
                              shuffle = True,
                              random_state = seeding_constant                                      
                            )

train_model_cnn_data = split_list_cnn[0]
test_model_cnn_data = split_list_cnn[1]
train_model_cnn_labels = split_list_cnn[2]
test_model_cnn_labels = split_list_cnn[3]

filter_count = 32
i=0
normalized_cnn_test_data = train_model_cnn_data/255

#Define the model with layers and hyperparams as per requiremnets
model_cnn_terrain = tf.keras.Sequential(layers=[
    tf.keras.layers.Conv2D(filter_count, kernel_size=(3,3), activation=tf.nn.leaky_relu),    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filter_count*(2**(i+1)), kernel_size=(3,3), activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filter_count*(2**(i+1)), kernel_size=(3,3), activation=tf.nn.relu, use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filter_count*(2**(i+2)), kernel_size=(3,3), activation=tf.nn.elu, use_bias=True),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(filter_count*(2**(i+4)), activation=tf.nn.elu, use_bias=True),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(len(np.unique(labels)), activation = tf.nn.softmax)])

#Add the optimizer, loss, metrics and finalize the configuration
model_cnn_terrain.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

#Train the model with train data set and include validation to see the accuracy
history_cnn_1 = model_cnn_terrain.fit(train_model_cnn_data,
                                train_model_cnn_labels,
                                epochs = 8,
                                validation_split = 0.1,
                                verbose = 1
                                )
