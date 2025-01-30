import tensorflow as tf
from src.config import IMG_SIZE, BATCH_SIZE
from src.preprocessing import create_dataset
from src.models.model import create_model

# Initialize components
train_dataset = create_dataset('data/train', BATCH_SIZE)
model = create_model()

# Start training
model.fit(train_dataset, epochs=10)
