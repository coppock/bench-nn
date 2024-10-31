all: resnet.engine bert.engine

resnet.engine: resnet.onnx
	trt/build -oresnet.engine resnet.onnx

resnet.onnx:
	curl https://zenodo.org/records/4735647/files/resnet50_v1.onnx \
	    >resnet.onnx

bert.engine: bert.onnx
	trt/build -obert.engine bert.onnx

bert.onnx:
	curl https://zenodo.org/records/3733910/files/model.onnx >bert.onnx
