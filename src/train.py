import tensorflow as tf
from config import BATCH_SIZE
from preprocessing import create_dataset
from models.model import create_model

# Initialize components
train_dataset = create_dataset("data/train", BATCH_SIZE)
model = create_model()

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Start training
history = model.fit(train_dataset, epochs=50, callbacks=[early_stopping])
