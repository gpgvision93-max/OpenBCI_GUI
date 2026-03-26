import torch
from transformers import GPT2TokenizerFast

from newfile import TransformerConfig, Transformer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TransformerConfig()
    model = Transformer(cfg).to(device)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out_ids = model.generate(inputs, max_new_tokens=50, temperature=1.0, top_k=50)

    generated_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    print("Generated:\n", generated_text)


if __name__ == "__main__":
    main()
