import tensorflow as tf

# Step 1: Load your trained model
model = tf.keras.models.load_model("action.h5")

# Step 2: Ensure the model has a fixed input shape (Important for LSTMs)
fixed_input_shape = (30, 1662)  # Adjust these values based on your model
new_model = tf.keras.Sequential()
new_model.add(tf.keras.layers.Input(shape=fixed_input_shape))

# Copy layers from original model
for layer in model.layers:
    new_model.add(layer)

# Save the new fixed-shape model
new_model.save("action.h5")

# Step 3: Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)

# Fix 1: Enable resource variable support
converter.experimental_enable_resource_variables = True  

# Fix 2: Allow TensorFlow ops in TFLite if required
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite operations
    tf.lite.OpsSet.SELECT_TF_OPS     # Allow non-convertible TensorFlow ops
]

# Fix 3: Disable experimental lowering of tensor list ops
converter._experimental_lower_tensor_list_ops = False  

# Convert the model
tflite_model = converter.convert()

# Step 4: Save the TFLite model
with open("action.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TFLite!")