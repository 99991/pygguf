# pygguf

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser in Python with NumPy-vectorized dequantization for `Q4_K` and `Q6_K` GGML tensors.

#### DISCLAIMER

For now, I have only implemented a small subset of the GGUF file format (namely the subset required to load TinyLlama with `Q4_K_M` or `Q8_0` quantization).
Also, I have barely tested this. It might not work correctly.
The mean squared error to the TinyLlama safetensors model is small, but the shapes are transposed for some reason.
I still have to look into that.
But there is little in-depth documentation of the GGUF file format at the time of writing, so I figured that this incomplete code might still be useful to some.

# Prerequisites

Install NumPy:

```bash
pip install numpy
```

Download the `Q4_K_M` model file from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main

# Example

```python
import gguf

filename = "data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

with open(filename, "rb") as f:
    # Load metadata
    info, tensorinfo = gguf.load_gguf(f)

    # Print metadata
    for key, value in info.items():
        print(f"{key:30} {repr(value)[:100]}")

    # Load tensors
    for name in tensorinfo:
        weights = gguf.load_gguf_tensor(f, tensorinfo, name)

        print(name, type(weights), weights.shape)
```

# Testing

For testing, follow these steps:

1. Install required libraries (only required for testing)
    * `pip install safetensors`
2. Create the directory `data`
3. Create the subdirectories `TinyLlama-1.1B-Chat-v1.0` and `TinyLlama-1.1B-Chat-v1.0-GGUF`
4. Download `model.safetensors` from https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main into the first subdirectory.
5. Download `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` and `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main into the second subdirectory.

It should look like this:

```
data
├── TinyLlama-1.1B-Chat-v1.0
│   └── model.safetensors
└── TinyLlama-1.1B-Chat-v1.0-GGUF
    ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    └── tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

On Linux, you can do this with:

```
mkdir -p data/TinyLlama-1.1B-Chat-v1.0
mkdir -p data/TinyLlama-1.1B-Chat-v1.0-GGUF
wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true' -O 'data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true' -O 'data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf'
wget 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors?download=true' -O 'data/TinyLlama-1.1B-Chat-v1.0/model.safetensors'
```

Finally, run the tests:

```bash
python test.py
```
