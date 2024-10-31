all: resnet.engine bert.engine gptj6b/engine

resnet.engine: resnet.onnx
	trt/build -oresnet.engine resnet.onnx

resnet.onnx:
	curl https://zenodo.org/records/4735647/files/resnet50_v1.onnx \
	    >resnet.onnx

bert.engine: bert.onnx
	trt/build -obert.engine bert.onnx

bert.onnx:
	curl https://zenodo.org/records/3733910/files/model.onnx >bert.onnx

gptj6b/engine: gptj6b/checkpoint
	trtllm-build --checkpoint_dir=gptj6b/checkpoint --gemm_plugin=float16 \
	    --max_batch_size=16 --remove_input_padding=enable \
	    --output_dir=gptj6b/engine
	touch gptj6b/engine

gptj6b/checkpoint:
	python third-party/TensorRT-LLM/examples/gptj/convert_checkpoint.py \
	    --model_dir=third-party/gpt-j-6b --output_dir=gptj6b/checkpoint
	touch gptj6b/checkpoint
