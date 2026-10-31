from pathlib import Path

import matplotlib.pyplot as plt
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, output_dir="attention_maps"):
        """Check attention probabilities and plot the first and last layers."""
        word_ids = self.tokenizer.encode(sentence)[:block_size]
        if not word_ids:
            raise ValueError("The sentence must contain at least one token")

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([word_ids], dtype=torch.long, device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _, attention_maps = self.model(input_tensor)
        finally:
            self.model.train(was_training)

        for layer, attention in enumerate(attention_maps, start=1):
            row_sums = attention.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                raise ValueError(f"Attention rows do not sum to 1 in layer {layer}")
        print("Attention rows sum to 1 in every layer and head.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokens = [self.tokenizer.itos[word_id] for word_id in word_ids]
        paths = []
        for layer_index in sorted({0, len(attention_maps) - 1}):
            # Plot the first head from each selected layer.
            attention = attention_maps[layer_index][0, 0].cpu().numpy()
            fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.4), 6))
            heatmap = ax.imshow(attention, cmap="hot", vmin=0, vmax=1)
            ax.set_xticks(range(len(tokens)), labels=tokens, rotation=90)
            ax.set_yticks(range(len(tokens)), labels=tokens)
            ax.set_xlabel("Key token")
            ax.set_ylabel("Query token")
            ax.set_title(f"Layer {layer_index + 1}, head 1")
            fig.colorbar(heatmap, ax=ax, label="Attention probability")
            fig.tight_layout()
            path = output_dir / f"attention_layer_{layer_index + 1}_head_1.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
            print(f"Saved attention plot: {path}")
        return paths
