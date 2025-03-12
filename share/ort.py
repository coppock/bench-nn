from pathlib import Path

from common import *

parser = ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

# This should happen before other importations in order to handle stdout abuse.
iter = Iterator(args.iteration_count)

import onnx
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(args.model, providers=providers)

# For some reason, latencies are bimodal when plain numpy arrays are used: use
# IO bindings instead.
dtypes = {
    'tensor(float)': onnx.TensorProto.FLOAT,
}
io_binding = session.io_binding()
for input in session.get_inputs():
    value = ort.OrtValue.ortvalue_from_shape_and_type(
        input.shape,
        dtypes[input.type],
        'cpu',
    )
    io_binding.bind_ortvalue_input(input.name, value)
for output in session.get_outputs():
    io_binding.bind_output(output.name, 'cpu')

for _ in iter:
    session.run_with_iobinding(io_binding)
