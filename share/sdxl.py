from cuda import cudart

from common import *

parser = ArgumentParser()
parser.add_argument('--height', default=512, type=int)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--framework-model-dir')
parser.add_argument('--onnx-dir')
parser.add_argument('--engine-dir')
args = parser.parse_args()

iter = Iterator(args.iteration_count)

from demo_txt2img_xl import StableDiffusionXLPipeline

demo = StableDiffusionXLPipeline(vae_scaling_factor=0.13025, version='xl-1.0',
                                 nvtx_profile=False,
                                 framework_model_dir=args.framework_model_dir)
demo.loadEngines(args.framework_model_dir, args.onnx_dir, args.engine_dir,
                 onnx_opset=19, opt_batch_size=args.batch_size,
                 opt_image_height=args.height, opt_image_width=args.width)

_, shared_device_memory = cudart.cudaMalloc(demo.get_max_device_memory())
demo.activateEngines(shared_device_memory)
demo.loadResources(args.height, args.width, args.batch_size, None)

for _ in iter:
    demo.run(['astronaut riding a horse on mars'], [''], args.height,
             args.width, args.batch_size, 1, 0, False)

demo.teardown()
