all: resnet.engine bert.engine

resnet.engine: resnet.onnx
	./trt -s'"input_tensor:0":1x3x224x224,1x3x224x224,8x3x224x224' \
	    -oresnet.engine resnet.onnx

resnet.onnx:
	curl https://zenodo.org/records/4735647/files/resnet50_v1.onnx \
	    >resnet.onnx

bert.engine: bert.onnx
	./trt -sinput_ids:1x384,1x384,8x384 -sinput_mask:1x384,1x384,8x384 \
	    -ssegment_ids:1x384,1x384,8x384 -obert.engine bert.onnx

bert.onnx:
	curl https://zenodo.org/records/3733910/files/model.onnx >bert.onnx
