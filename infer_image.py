import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
import datetime


def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')

    image = image.resize((input_shape[2], input_shape[3]))

    image_array = np.array(image).astype(np.float32)

    # Normalize the image (assuming model expects values in range [-1, 1])
    image_array = image_array / 255.0
    image_array = image_array * 2.0 - 1.0

    # Transpose to match model's input shape (N, C, H, W)
    image_array = image_array.transpose(2, 0, 1)

    image_array = image_array.reshape(1, 3, input_shape[2], input_shape[3])

    return image_array

def run_inference(model_path, image_array):
    session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name
    print("Input name:", input_name)

    inputs = {input_name: image_array}

    output_names = [output.name for output in session.get_outputs()]
    print("Output names:", output_names)

    try:
        for output_name in output_names:
            outputs = session.run([output_name], inputs)
            print(f"Output ({output_name}): shape {outputs[0].shape}, values {outputs[0]}")
            inputs = {output_name: outputs[0]}

    except Exception as e:
        print(f"Error during inference: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python infer.py <model.onnx> <image.jpg>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    session = ort.InferenceSession(model_path)
    input_shape = session.get_inputs()[0].shape
    print(f"Model expects input shape: {input_shape}")

    image_array = preprocess_image(image_path, input_shape)
    print(f"Preprocessed input shape: {image_array.shape}")

    if image_array.shape != tuple(input_shape):
        print(f"Error: Processed input shape {image_array.shape} does not match expected shape {tuple(input_shape)}")
        sys.exit(1)

    print("Starting at: ", datetime.datetime.now())
    run_inference(model_path, image_array)
    print("Finished at: ", datetime.datetime.now())

if __name__ == "__main__":
    main()
