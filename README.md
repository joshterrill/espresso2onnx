# espresso2onnx

## Converting espresso files

Locate the folder that contains the following files:
* `*.espresso.net`
* `*.espresso.weights`
* `*.espresso.shape`

i.e. `model.espresso.net`, `model.espresso.weights`, `model.espresso.shape`
or `model_fp16.espresso.net`, `model_fp16.espresso.weights`, `model_fp16.espresso.shape`

```bash
python espresso2onnx.py /path/to/espresso/model/directory
# model.onnx file is saved to model directory
```

## Run inference on `model.onnx`

Image-based models:

```bash
python infer_image.py /path/to/model.onnx /path/to/image.png
```