from pathlib import Path

from common import *

parser = ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

# This should happen before other importations in order to handle stdout abuse.
iter = Iterator(args.iteration_count)

import numpy as np
from polygraphy import util
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from polygraphy.backend.common import BytesFromPath

def runner(model):
    match Path(model).suffix:
        case '.engine' | '.plan':
            return TrtRunner(EngineFromBytes(BytesFromPath(model)))
        case '.onnx':
            return OnnxrtRunner(SessionFromOnnx(model, ['cuda']))

with runner(args.model) as r:
    input_metadata = {name: (
        dtype.numpy(),
        util.override_dynamic_shape(
            shape,
            default_shape_value=args.batch_size,
        ),
    ) for name, (dtype, shape) in r.get_input_metadata(
        use_numpy_dtypes=False,
    ).items()}
    for _ in iter:
        feed_dict = {name: np.empty(shape, dtype=dtype)
                        for name, (dtype, shape) in input_metadata.items()}
        r.infer(feed_dict)
