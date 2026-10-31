import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Mix each token's representation with the other tokens in the sequence."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.output = nn.Linear(n_embd, n_embd)

    def forward(self, x, padding_mask=None):
        batch_size, seq_len, n_embd = x.shape

        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        queries = split_heads(self.query(x))
        keys = split_heads(self.key(x))
        values = split_heads(self.value(x))

        scores = queries @ keys.transpose(-2, -1) * self.head_dim ** -0.5
        if padding_mask is not None:
            # True marks a padded key, which no token should attend to.
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = attention @ values
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.output(context), attention

class EncoderBlock(nn.Module):
    """One self-attention layer followed by a position-wise feedforward layer."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        self.attention_norm = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.feedforward_norm = nn.LayerNorm(n_embd)

    def forward(self, x, padding_mask=None):
        attended, attention_map = self.attention(x, padding_mask)
        x = self.attention_norm(x + attended)
        x = self.feedforward_norm(x + self.feedforward(x))
        return x, attention_map


class Encoder(nn.Module):
    """Turn token IDs into contextual vectors using Transformer encoder blocks."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, pad_id=0):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [EncoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )

    def forward(self, token_ids):
        _, seq_len = token_ids.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence exceeds the encoder's block_size")

        padding_mask = token_ids == self.pad_id
        if padding_mask.all(dim=1).any():
            raise ValueError("Each sequence must contain at least one non-padding token")

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_maps = []
        for block in self.blocks:
            x, attention_map = block(x, padding_mask)
            attention_maps.append(attention_map)
        return x, attention_maps



class Classifier(nn.Module):
    """Classify a speech segment from the mean of its encoder outputs."""

    def __init__(self, encoder, n_hidden, n_output=3):
        super().__init__()
        self.encoder = encoder
        n_embd = encoder.token_embedding.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, token_ids):
        vectors, _ = self.encoder(token_ids)
        valid = (token_ids != self.encoder.pad_id).unsqueeze(-1)
        mean_vector = (vectors * valid).sum(dim=1) / valid.sum(dim=1)
        return self.layers(mean_vector)