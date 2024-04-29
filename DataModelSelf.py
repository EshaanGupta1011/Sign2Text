from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Create ImageDataGenerator for both training and validation datasets
datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)

# Load images from the same directory for both training and validation
train_dataset = datagen.flow_from_directory("./Data/",
                                            target_size=(450, 450),
                                            batch_size=19,
                                            class_mode="categorical",
                                            subset="training")

validation_dataset = datagen.flow_from_directory("./Data/",
                                                 target_size=(450, 450),
                                                 batch_size=19,
                                                 class_mode="categorical",
                                                 subset="validation")


# Create the model architecture with dropout
model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(450, 450, 3)),
    MaxPool2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPool2D(2, 2),
    Dropout(0.5),  # Add dropout layer
    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(24, activation="softmax")
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Fit the model
model.fit(
    train_dataset,
    steps_per_epoch=37,
    epochs=25,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

model.save("./Model/model.h5")