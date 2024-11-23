all: resnet.engine bert.engine gptj/engine retinanet.onnx 3dunet.engine sdxl

resnet.engine: resnet.onnx
	polygraphy convert -oresnet.engine resnet.onnx \
	    --trt-min-shapes input_tensor:0:[1,3,224,224] \
	    --trt-opt-shapes input_tensor:0:[1,3,224,224] \
	    --trt-max-shapes input_tensor:0:[16,3,224,224]

resnet.onnx:
	curl https://zenodo.org/records/4735647/files/resnet50_v1.onnx \
	    >resnet.onnx

bert.engine: bert.onnx
	polygraphy convert -obert.engine bert.onnx --trt-min-shapes \
	    input_ids:[1,384] input_mask:[1,384] segment_ids:[1,384] \
	    --trt-opt-shapes input_ids:[1,384] input_mask:[1,384] \
	    segment_ids:[1,384] --trt-max-shapes input_ids:[16,384] \
	    input_mask:[16,384] segment_ids:[16,384]

bert.onnx:
	curl https://zenodo.org/records/3733910/files/model.onnx >bert.onnx

gptj/engine: gptj/checkpoint
	trtllm-build --checkpoint_dir=gptj/checkpoint --gemm_plugin=float16 \
	    --max_batch_size=16 --remove_input_padding=enable \
	    --output_dir=gptj/engine
	touch gptj6b/engine

gptj/checkpoint:
	python third-party/TensorRT-LLM/examples/gptj/convert_checkpoint.py \
	    --model_dir=third-party/gpt-j-6b --output_dir=gptj/checkpoint
	touch gptj/checkpoint

retinanet.onnx: retinanet.pth
	PYTHONPATH=third-party/training/single_stage_detector/ssd \
	    third-party/training/single_stage_detector/scripts/pth_to_onnx.py \
	    --input=retinanet.pth --output=retinanet.pth --image-size 800 800

retinanet.pth: retinanet.zip
	unzip retinanet.zip

retinanet.zip:
	curl https://zenodo.org/records/6605272/files/retinanet_model_10.zip \
	    >retinanet.zip

3dunet.engine: 3dunet.onnx
	polygraphy convert -o3dunet.engine 3dunet.onnx \
	    --trt-min-shapes input:[1,1,128,128,128] \
	    --trt-opt-shapes input:[1,1,128,128,128] \
	    --trt-max-shapes input:[8,1,128,128,128]

3dunet.onnx:
	curl https://zenodo.org/records/5597155/files/3dunet_kits19_128x128x128_dynbatch.onnx \
	    >3dunet.onnx

sdxl:
	../bin/sdxl -m1
	touch sdxl
