"""Model definition for defect detection."""

# Standard library imports
import logging
from pathlib import Path
from typing import Union

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import custom_object_scope

logger = logging.getLogger(__name__)

class CastToFloat32(Layer):
    """Cast inputs to float32."""
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

class SubtractLayer(Layer):
    """Subtract a constant value."""
    
    def __init__(self, value=127.5, **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    def call(self, inputs):
        return inputs - self.value
    
    def get_config(self):
        config = super().get_config()
        config.update({"value": self.value})
        return config

class DivideLayer(Layer):
    """Divide by a constant value."""
    
    def __init__(self, value=127.5, **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    def call(self, inputs):
        return inputs / self.value
    
    def get_config(self):
        config = super().get_config()
        config.update({"value": self.value})
        return config

class PreprocessingLayer(Layer):
    """Combined preprocessing layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cast = CastToFloat32()
        self.subtract = SubtractLayer(127.5)
        self.divide = DivideLayer(127.5)
    
    def call(self, inputs):
        x = self.cast(inputs)
        x = self.subtract(x)
        x = self.divide(x)
        return x
    
    def get_config(self):
        return super().get_config()

class TrueDivide(Layer):
    """Legacy TrueDivide layer."""
    
    def __init__(self, divisor=None, **kwargs):
        super().__init__(**kwargs)
        self.divisor = divisor
        self._has_positional_arg = False
    
    def build(self, input_shape):
        """Build layer."""
        super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            inputs: Input tensor or tuple of (tensor, divisor)
            **kwargs: Additional arguments
        """
        if isinstance(inputs, (list, tuple)):
            x, y = inputs
            return tf.cast(x, tf.float32) / tf.cast(y, tf.float32)
        elif self.divisor is not None:
            return tf.cast(inputs, tf.float32) / tf.cast(self.divisor, tf.float32)
        elif 'divisor' in kwargs:
            return tf.cast(inputs, tf.float32) / tf.cast(kwargs['divisor'], tf.float32)
        return tf.cast(inputs, tf.float32)
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({"divisor": self.divisor})
        return config

class DefectDetectionModel:
    """Neural network model for defect detection."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the model.

        Args:
            num_classes: Number of output classes
        """
        # Create input layer
        inputs = Input(shape=(224, 224, 3), name='input_1')
        
        # Add preprocessing layer
        x = PreprocessingLayer(name='preprocessing')(inputs)
        
        # Create base model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers
        x = base_model(x)
        x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = BatchNormalization(name='batch_normalization')(x)
        x = Dense(1024, activation='relu', name='dense')(x)
        x = Dropout(0.5, name='dropout')(x)
        predictions = Dense(num_classes, activation='softmax', name='dense_1')(x)
        
        self.model = Model(
            inputs=inputs,
            outputs=predictions,
            name='defect_detection_model'
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model.predict(x, verbose=0)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: Union[str, Path]
    ) -> "DefectDetectionModel":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        try:
            logger.info(f"Loading model from {checkpoint_path}")
            
            # Create model instance
            instance = cls()
            
            # Define custom objects
            custom_objects = {
                'TrueDivide': TrueDivide,
                'PreprocessingLayer': PreprocessingLayer,
                'CastToFloat32': CastToFloat32,
                'SubtractLayer': SubtractLayer,
                'DivideLayer': DivideLayer
            }
            
            # Load model with custom_object_scope
            with custom_object_scope(custom_objects):
                loaded_model = load_model(str(checkpoint_path), compile=False)
                instance.model = loaded_model
            
            logger.info("Model loaded successfully")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def eval(self):
        """Set model to evaluation mode (no-op for TensorFlow)."""
        pass  # Not needed for TensorFlow/Keras models
