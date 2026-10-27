"""
backend/services/gradcam_service.py — Grad-CAM heatmap generation.
Compatible with custom CNN and ResNet50 classification models.
NOT applicable to Random Forest.
"""
import logging
import base64
from typing import Optional
import numpy as np
import cv2
import tensorflow as tf

logger = logging.getLogger(__name__)


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,      # preprocessed (1, H, W, 3)
    class_idx: int,
    last_conv_layer_name: str,
    original_img_array: Optional[np.ndarray] = None,  # raw (H, W, 3) for overlay
) -> Optional[str]:
    """
    Generate Grad-CAM heatmap and return as base64 JPEG string.
    
    Args:
        model: Keras model (CNN or ResNet)
        img_array: preprocessed batch array (1, H, W, 3)
        class_idx: integer class index for target class
        last_conv_layer_name: name of the last convolutional layer
        original_img_array: optional raw image for overlay (uint8, RGB)
    
    Returns:
        Base64-encoded JPEG string, or None if Grad-CAM fails.
    """
    try:
        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tf.cast(img_array, tf.float32))
            loss = predictions[:, class_idx]

        # Compute gradient of loss w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)            # (1, h, w, C)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        conv_outputs = conv_outputs[0]                         # (h, w, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap = tf.squeeze(heatmap)                           # (h, w)

        # ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Determine original image size
        if original_img_array is not None:
            target_h, target_w = original_img_array.shape[:2]
        else:
            # Fallback: use input shape from img_array
            target_h, target_w = img_array.shape[1], img_array.shape[2]

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap_np, (target_w, target_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay on original image
        if original_img_array is not None:
            orig_bgr = cv2.cvtColor(original_img_array, cv2.COLOR_RGB2BGR)
        else:
            # Reconstruct from preprocessed input (CNN uses /255, so multiply back)
            raw = (img_array[0] * 255.0).clip(0, 255).astype(np.uint8)
            orig_bgr = cv2.cvtColor(cv2.resize(raw, (target_w, target_h)), cv2.COLOR_RGB2BGR)

        superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', superimposed, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return None
