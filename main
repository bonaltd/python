import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

def create_model():
    model = models.Sequential([
        # Input layer - expects images of size 224x224 with 3 color channels
        layers.Input(shape=(224, 224, 3)),
        
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the output for dense layers
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to prevent overfitting
        layers.Dense(2, activation='softmax')  # 2 outputs: negative or positive
    ])
    
    return model

def preprocess_image(image_path):
    # Load and preprocess a single image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to expected size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch
    return img_array

def train_model(model, train_data_dir):
    # Set up data augmentation for training
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Set up the data generators
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    return history

def predict_result(model, image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Get the result
    result = "Positive" if np.argmax(prediction[0]) == 1 else "Negative"
    confidence = float(np.max(prediction[0]))
    
    return result, confidence

if __name__ == "__main__":
    # Create and train the model
    model = create_model()
    
    # You would need to specify your training data directory
    train_data_dir = "path/to/your/training/data"
    
    # Train the model
    history = train_model(model, train_data_dir)
    
    # Save the model
    model.save('hiv_test_model.h5')
    
    # Example prediction
    test_image_path = "path/to/test/image.jpg"
    result, confidence = predict_result(model, test_image_path)
    print(f"Test Result: {result}")
    print(f"Confidence: {confidence:.2%}")
