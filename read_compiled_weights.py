import os
import struct
import sys

def read_compiled_weights(mlmodelc_path):
    """Read a compiled model.espresso.weights file.

    Args:
        mlmodelc_path (str): location of mlmodelc folder.

    Returns: dict[int, list[float]] of section to list of weights.
    """
    layer_bytes = []
    layer_data = {}
    filename = os.path.join(mlmodelc_path, 'model.espresso.weights')
    with open(filename, 'rb') as f:
        # First byte of the file is an integer with how many
        # sections there are.  This lets us iterate through each section
        # and get the map for how to read the rest of the file.
        num_layers = struct.unpack('<i', f.read(4))[0]

        f.read(4)  # padding bytes

        # The next section defines the number of bytes each layer contains.
        # It has a format of
        # | Layer Number | <padding> | Bytes in layer | <padding> |
        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))

        # Read actual layer weights.  Weights are floats as far as I can tell.
        for layer_num, num_bytes in layer_bytes:
            print(layer_num, num_bytes)
            byte_data = f.read(num_bytes)
                
            # Calculate the number of floats
            num_floats = num_bytes // 4

            # Unpack the data
            data = struct.unpack('f' * num_floats, byte_data)
            layer_data[layer_num] = data

        return layer_data
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python read_compiled_weights.py <path to coreml folder>")
        sys.exit(1)
    mlmodelc_path = sys.argv[1]
    weights = read_compiled_weights(mlmodelc_path)
    print(weights)