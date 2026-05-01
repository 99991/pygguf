# pygguf

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser in Python with NumPy-vectorized dequantization of GGML tensors.

#### DISCLAIMER

* ⚠️ This library is not being maintained anymore, since llama.cpp has added support for loading GGUF files from Python. It merely serves historic and educational purposes now. You are probably looking for the [gguf library](https://pypi.org/project/gguf/) instead.
* This code has only been tested for the TinyLlama model. It might (or might not) work for other models.
If any issues arise, a probable source might be the weird transposition of the key and query weights.

# Prerequisites

Install NumPy:

```bash
pip install numpy
```

Download the `Q4_K_M` model file from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main

```bash
mkdir -p 'data/TinyLlama-1.1B-Chat-v1.0-GGUF'
wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true' -O 'data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
```

Install pygguf:

```bash
git clone https://github.com/99991/pygguf.git
cd pygguf
pip install -e .
```

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
    * `pip install tqdm requests safetensors`
2. Run
    * `python test.py`
    * This will download the TinyLlama model (safentesors, GGUF) from
        * https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
        * https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
