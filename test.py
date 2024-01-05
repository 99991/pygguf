import gguf
import numpy as np

def translate_name(name):
    # Translate names from GGUF model to safetensors model.
    if name == "output.weight":
        return "lm_head.weight"

    if name == "token_embd.weight":
        return "model.embed_tokens.weight"

    if name == "output_norm.weight":
        return "model.norm.weight"

    name = name.replace("blk.", "model.layers.")
    name = name.replace(".attn_norm.weight", ".input_layernorm.weight")
    name = name.replace(".ffn_down.weight", ".mlp.down_proj.weight")
    name = name.replace(".ffn_gate.weight", ".mlp.gate_proj.weight")
    name = name.replace(".ffn_up.weight", ".mlp.up_proj.weight")
    name = name.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
    name = name.replace(".attn_q.weight", ".self_attn.q_proj.weight")
    name = name.replace(".attn_k.weight", ".self_attn.k_proj.weight")
    name = name.replace(".attn_v.weight", ".self_attn.v_proj.weight")
    name = name.replace(".attn_output.weight", ".self_attn.o_proj.weight")

    return name

def main():
    import time
    from safetensors.torch import load_file

    # Load safetensors model to compare against
    state_dict = load_file("data/TinyLlama-1.1B-Chat-v1.0/model.safetensors")

    print("safetensors model for comparison")
    for key, value in state_dict.items():
        print(f"{key:30} {value.shape}")
    print()

    for filename in [
        "data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "data/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    ]:
        with open(filename, "r+b") as f:
            # also works with mmap (at least on Linux)
            #import mmap
            #f =  mmap.mmap(f.fileno(), 0)

            info, tensorinfo = gguf.load_gguf(f)

            print("gguf metadata")
            for key, value in info.items():
                print(f"{key:30} {repr(value)[:70]}")
            print()
            print("gguf tensors")
            for key, value in tensorinfo.items():
                print(f"{key:30} {str(value)[:70]}")
            print()

            for name in tensorinfo:
                start_time = time.perf_counter()

                weights = gguf.load_gguf_tensor(f, tensorinfo, name)

                shape = tensorinfo[name]["shape"]

                # For some reason, the key and query weights are transposed
                # in this weird way in the GGUF file. Not sure why.
                if ".attn_k." in name or ".attn_q." in name:
                    num_heads = info["llama.attention.head_count"]
                    tmp_shape = (shape[-1] // num_heads // 2, num_heads, 2, shape[0])
                    weights = weights.reshape(tmp_shape)
                    weights = weights.transpose(0, 2, 1, 3)
                    weights = weights.reshape(shape[::-1])

                other_name = translate_name(name)

                expected = state_dict[other_name].float().numpy().astype(np.float32)

                ms = (time.perf_counter() - start_time) * 1000

                mse = np.mean(np.square(weights - expected))

                ggml_type = tensorinfo[name]["ggml_type"]

                print(f"MSE {mse:.10f} {name:30} ggml_type {ggml_type:2} {str(shape):13} {ms:7.3f} ms")

                if "Q8_0" in filename:
                    assert mse < 2e-6
                else:
                    assert mse < 2e-5

    print("Tests passed :)")

if __name__ == "__main__":
    main()
