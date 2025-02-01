OPT = ../opt

all: resnet.engine bert.engine retinanet.onnx 3dunet.engine

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

retinanet.engine: retinanet.onnx
	polygraphy convert -oretinanet.engine retinanet.onnx

retinanet.onnx:
	curl https://zenodo.org/records/6617879/files/resnext50_32x4d_fpn.onnx \
	    >retinanet.onnx

3dunet.engine: 3dunet.onnx
	polygraphy convert -o3dunet.engine 3dunet.onnx \
	    --trt-min-shapes input:[1,1,128,128,128] \
	    --trt-opt-shapes input:[1,1,128,128,128] \
	    --trt-max-shapes input:[8,1,128,128,128]

3dunet.onnx:
	curl https://zenodo.org/records/5597155/files/3dunet_kits19_128x128x128_dynbatch.onnx \
	    >3dunet.onnx
