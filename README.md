## Requirements

* CUDA 12
* cuDNN 9

```sh
sudo apt update
sudo apt install -y libopenmpi-dev
```

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

## Build

```sh
make
```

## To Do

* [X] Implement _trt/llmexec_ with command-line parameters
  - Input and output sequence lengths
  - Batch size
* [ ] How does LC latency scale with BE batch size or input sequence length?
  - With multiple configurations
    + MPS
    + MPS with client priority
    + Alone
  - What model characteristics explain interference?
    + Grid launches per time
    + Grid duration
    + Grid launch _wave_ duration
    + Utilization of various resources
* More models
  - [X] RetinaNet
  - [X] 3D U-Net
  - [ ] Llama 2 70B
  - [ ] Mixtral 8x7B
  - [X] SDXL 1.0
  - [ ] DLRM-DCNv2
