
from webmnist.model import Model, LeNet5
from onnx_tf.backend import prepare
from tensorflowjs.converters.tf_saved_model_conversion_v2 import (
    convert_tf_saved_model,
)

import os
import torch
import onnx

def export(input_path: str, output_path: str) -> None:

    if os.path.isdir(input_path):
        input_path = f"{input_path}/best.pt"   

    onnx_path = f"{output_path}.onnx"
    tensorflow_path = f"{output_path}.pb"
    tensorflowjs_path = output_path

    # model = Model()
    model = LeNet5(10)
    model.load_state_dict(torch.load(input_path))
    model = model.eval()

    torch.onnx.export(
        model,
        torch.zeros((1, 1, 28, 28)),
        onnx_path,
        do_constant_folding=True,
        export_params=True,
        input_names=["img"],
        output_names=["preds"],
        opset_version=10,
        verbose=True,
    )

    model = onnx.load(onnx_path)
    prepare(model).export_graph(tensorflow_path)
    convert_tf_saved_model(tensorflow_path, tensorflowjs_path)

    print("exported model.")
