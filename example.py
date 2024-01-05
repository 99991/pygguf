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
