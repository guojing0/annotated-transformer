import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # Some imports

    import copy
    import math
    import time
    import torch
    import torch.nn as nn
    from torch.nn.functional import log_softmax
    from torch.optim.lr_scheduler import LambdaLR
    import pandas as pd
    import altair as alt

    return LambdaLR, alt, copy, log_softmax, math, nn, pd, time, torch


@app.cell
def _(torch):
    # Some convenience helper functions used throughout the notebook

    RUN_EXAMPLES = True

    def is_interactive_notebook():
        return __name__ == "__main__"

    def show_example(fn, args=[]):
        if __name__ == "__main__" and RUN_EXAMPLES:
            return fn(*args)

    def execute_example(fn, args=[]):
        if __name__ == "__main__" and RUN_EXAMPLES:
            fn(*args)


    class DummyOptimizer(torch.optim.Optimizer):
        def __init__(self):
            self.param_groups = [{"lr": 0}]
            None

        def step(self):
            None

        def zero_grad(self, set_to_none=False):
            None


    class DummyScheduler:
        def step(self):
            None


    return DummyOptimizer, DummyScheduler, show_example


@app.cell
def _(log_softmax, nn):
    ### Modular architecture

    class EncoderDecoder(nn.Module):
        """
        Standard encoder-decoder architecture.
        """

        def __init__(self, encoder, decoder, source_embed, target_embed, generator):
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.source_embed = source_embed
            self.target_embed = target_embed
            self.generator = generator

        def forward(self, source, target, source_mask, target_mask):
            "Take in and process masked source and target sequences."
            return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

        def encode(self, source, source_mask):
            return self.encoder(self.source_embed(source), source_mask)

        def decode(self, memory, source_mask, target, target_mask):
            return self.decoder(self.target_embed(target), memory, source_mask, target_mask)


    class Generator(nn.Module):
        "Define standard linear + softmax generation step."

        def __init__(self, d_model, vocab):
            super(Generator, self).__init__()
            self.proj = nn.Linear(d_model, vocab)

        def forward(self, x):
            return log_softmax(self.proj(x), dim=-1)


    return EncoderDecoder, Generator


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        "Construct a layer norm module."

        def __init__(self, features, eps=1e-6):
            super(LayerNorm, self).__init__()
            self.a_2 = nn.Parameter(torch.ones(features))
            self.b_2 = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


    return (LayerNorm,)


@app.cell
def _(copy, nn):
    def clones(module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


    return (clones,)


@app.cell
def _(LayerNorm, clones, nn):
    class Encoder(nn.Module):
        "Core encoder is a stack of N layers"

        def __init__(self, layer, N):
            super(Encoder, self).__init__()
            self.layers = clones(layer, N)
            self.norm = LayerNorm(layer.size)

        def forward(self, x, mask):
            "Pass the input and mask through each layer in turn."
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)


    return (Encoder,)


@app.cell
def _(LayerNorm, nn):
    class SubplayerConnection(nn.Module):
        """
        A residual connection followed by a layer norm.
        """

        def __init__(self, size, dropout):
            super(SubplayerConnection, self).__init__()
            self.norm = LayerNorm(size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, subplayer):
            "Apply residual connection to any sublayer with the same size."
            return x + self.dropout(subplayer(self.norm(x)))


    return (SubplayerConnection,)


@app.cell
def _(SubplayerConnection, clones, nn):
    class EncoderLayer(nn.Module):
        "Encoder is made up of self-attention and feed-forward networks."

        def __init__(self, size, self_attn, feed_forward, dropout):
            super(EncoderLayer, self).__init__()
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.sublayer = clones(SubplayerConnection(size, dropout), 2)
            self.size = size

        def forward(self, x, mask):
            "Follow encoder figure for connections."
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)


    return (EncoderLayer,)


@app.cell
def _(LayerNorm, clones, nn):
    class Decoder(nn.Module):
        "Generic N-layer decoder with masking."

        def __init__(self, layer, N):
            super(Decoder, self).__init__()
            self.layers = clones(layer, N)
            self.norm = LayerNorm(layer.size)

        def forward(self, x, memory, source_mask, target_mask):
            for layer in self.layers:
                x = layer(x, memory, source_mask, target_mask)
            return self.norm(x)


    return (Decoder,)


@app.cell
def _(SubplayerConnection, clones, nn):
    class DecoderLayer(nn.Module):
        "Decoder is made up of self-attention, source-attention, and feed-forward networks."

        def __init__(self, size, self_attn, source_attn, feed_forward, dropout):
            super(DecoderLayer, self).__init__()
            self.size = size
            self.self_attn = self_attn
            self.source_attn = source_attn
            self.feed_forward = feed_forward
            self.sublayer = clones(SubplayerConnection(size, dropout), 3)

        def forward(self, x, memory, source_mask, target_mask):
            "Follow right figure for connections."
            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
            x = self.sublayer[1](x, lambda x: self.source_attn(x, m, m, source_mask))
            return self.sublayer[2](x, self.feed_forward)


    return (DecoderLayer,)


@app.cell
def _(torch):
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0


    return (subsequent_mask,)


@app.cell
def _(math, torch):
    ### Attention!

    def attention(query, key, value, mask=None, dropout=None):
        "Scaled dot production attention"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


    return (attention,)


@app.cell
def _(attention, clones, nn):
    class MultiHeadedAttention(nn.Module):

        def __init__(self, h, d_model, dropout=0.1):
            "Take in model size and number of heads"
            super(MultiHeadedAttention, self).__init__()
            assert d_model % h == 0
            # We assume d_v equal to d_k
            self.d_k = d_model // h
            self.h = h
            self.linears = clones(nn.Linear(d_model, d_model), 4)
            self.attn = None
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, value, mask=None):
            "Implement Fig 2"
            if mask is not None:
                # Same mask applied to all h heads.
                mask = mask.unsqueeze(1)

            nbatches = query.size(0)

            # 1. Do all the linear projections in batch from d_model => h * d_k
            query, key, value = [
                lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for lin, x in zip(self.linears, (query, key, value))
            ]

            # 2. Apply attention on all the projected vectors in batch
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

            # 3. "Concat" using a view and apply a final linear.
            x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))

            del query
            del key
            del value

            return self.linears[-1](x)


    return (MultiHeadedAttention,)


@app.cell
def _(nn):
    class PositionwiseFeedForward(nn.Module):
        "Implement FFN equation."

        def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.w_2(self.dropout(self.w_1(x).relu()))


    return (PositionwiseFeedForward,)


@app.cell
def _(math, nn):
    class Embeddings(nn.Module):

        def __init__(self, d_model, vocab):
            super(Embeddings, self).__init__()
            self.lut = nn.Embedding(vocab, d_model)
            self.d_model = d_model

        def forward(self, x):
            return self.lut(x) * math.sqrt(self.d_model)


    return (Embeddings,)


@app.cell
def _(math, nn, torch):
    class PositionalEncoding(nn.Module):

        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
            return self.dropout(x)


    return (PositionalEncoding,)


@app.cell
def _(
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    Generator,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    copy,
    nn,
):
    # Full model

    def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Consturct a model from hyperparameters."

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
            Generator(d_model, target_vocab),
        )

        # Important from original code from paper
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model


    return (make_model,)


@app.cell
def _(make_model, show_example, subsequent_mask, torch):
    # Inference

    def inference_test():
        test_model = make_model(11, 11, 2)
        test_model.eval()
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        src_mask = torch.ones(1, 1, 10)

        memory = test_model.encode(src, src_mask)
        ys = torch.zeros(1, 1).type_as(src)

        for i in range(9):
            out = test_model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = test_model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )

        print("Example Untrained Model Prediction:", ys)


    def run_tests():
        for _ in range(10):
            inference_test()


    show_example(run_tests)
    return


@app.cell
def _(subsequent_mask):
    # Training

    class Batch:
        """Object for holding a batch of data with mask during training."""

        def __init__(self, src, target=None, pad=2): # 2 = <blank>
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)

            if target is not None:
                self.target = target[:, :-1]
                self.target_y = target[:, 1:]
                self.target_mask = self.make_std_mask(self.target, pad)
                self.ntokens = (self.target_y != pad).data.sum()

        @staticmethod
        def make_std_mask(target, pad):
            "Create a mask to hide padding and future words."
            target_mask = (target != pad).unsqueeze(-2)
            target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask.data)
            return target_mask

    # Training loop

    class TrainState:
        """Track number of steps, examples, and token processed."""

        step: int = 0        # steps in current epoch
        accum_step: int = 0  # Number of gradient accumulation steps
        samples: int = 0     # Total number of examples used
        tokens: int = 0      # Total number of tokens processed

    return Batch, TrainState


@app.cell
def _(TrainState, time):
    def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState()
    ):
        """Train a single epoch."""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0

        for i, batch in enumerate(data_iter):
            out = model.forward(
                batch.src, batch.target, batch.src_mask, batch.target_mask
            )
            loss, loss_node = loss_compute(out, batch.target_y, batch.ntokens)

            if mode == "train" or mode == "train+log":
                loss_node.backward()

                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens

                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 40 == 1 and (mode == "train" or mode == "train+log"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0

            del loss
            del loss_node

        return total_loss / total_tokens, train_state

    return (run_epoch,)


@app.function
def rate(step, model_size, factor, warmup):
    """
    We have to default the step to 1 for LambdaLR function to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


@app.cell
def _(nn, torch):
    # Regularization

    ## Label smoothing

    ### This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score

    class LabelSmoothing(nn.Module):
        "Implement label smoothing."

        def __init__(self, size, padding_idx, smoothing=0.0):
            super(LabelSmoothing, self).__init__()
            self.criterion = nn.KLDivLoss(reduction="sum")
            self.padding_idx = padding_idx
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.size = size
            self.true_dist = None

        def forward(self, x, target):
            assert x.size(1) == self.size
            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            return self.criterion(x, true_dist.clone().detach())

    return (LabelSmoothing,)


@app.cell
def _(LabelSmoothing, alt, pd, show_example, torch):
    def example_label_smoothing():
        crit = LabelSmoothing(5, 0, 0.4)
        predict = torch.FloatTensor(
            [
                [0, 0.2, 0.7, 0.1, 0],
                [0, 0.2, 0.7, 0.1, 0],
                [0, 0.2, 0.7, 0.1, 0],
                [0, 0.2, 0.7, 0.1, 0],
                [0, 0.2, 0.7, 0.1, 0],
            ]
        )
        crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
        LS_data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "target distribution": crit.true_dist[x, y].flatten(),
                        "columns": y,
                        "rows": x,
                    }
                )
                for y in range(5)
                for x in range(5)
            ]
        )

        return (
            alt.Chart(LS_data)
            .mark_rect(color="Blue", opacity=1)
            .properties(height=200, width=200)
            .encode(
                alt.X("columns:O", title=None),
                alt.Y("rows:O", title=None),
                alt.Color(
                    "target distribution:Q", scale=alt.Scale(scheme="viridis")
                ),
            )
            .interactive()
        )


    show_example(example_label_smoothing)
    return


@app.cell
def _(LabelSmoothing, alt, pd, show_example, torch):
    def loss(x, crit):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
        predict = predict.clamp_min(1e-12)
        return crit(predict.log(), torch.LongTensor([1])).data

    def penalization_visualization():
        crit = LabelSmoothing(5, 0, 0.1)
        loss_data = pd.DataFrame(
            {
                "Loss": [loss(x, crit) for x in range(1, 100)],
                "Steps": list(range(99)),
            }
        ).astype("float")

        return (
            alt.Chart(loss_data)
            .mark_line()
            .properties(width=350)
            .encode(
                x="Steps",
                y="Loss",
            )
            .interactive()
        )

    show_example(penalization_visualization)
    return


@app.cell
def _(Batch, torch):
    # A first example

    ## Synthetic data

    def data_gen(V, batch_size, nbatches, device="cpu"):
        "Generate random data for a source-target copy task."
        for i in range(nbatches):
            data = torch.randint(1, V, size=(batch_size, 10), device=device)
            data[:, 0] = 1
            source = data.requires_grad_(False).clone().detach()
            target = data.requires_grad_(False).clone().detach()
            yield Batch(source, target, 0)

    return (data_gen,)


@app.class_definition
# Loss computation

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            ) / norm
        )

        return sloss.data * norm, sloss


@app.cell
def _(subsequent_mask, torch):
    # Greedy decoding

    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        memory = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

        for i in range(max_len - 1):
            out = model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )

        return ys

    return (greedy_decode,)


@app.cell
def _(
    DummyOptimizer,
    DummyScheduler,
    LabelSmoothing,
    LambdaLR,
    data_gen,
    greedy_decode,
    make_model,
    run_epoch,
    torch,
):
    def example_simple_model():
        V = 11
        device = torch.device("mps")
        print(f"Using device: {device}")
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = make_model(V, V, N=2).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, model_size=model.source_embed[0].d_model, factor=1.0, warmup=400
            ),
        )

        batch_size = 80
        for epoch in range(20):
            model.train()
            run_epoch(
                data_gen(V, batch_size, 20, device=device),
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train",
            )
            model.eval()
            run_epoch(
                data_gen(V, batch_size, 5, device=device),
                model,
                SimpleLossCompute(model.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )[0]

        model.eval()
        src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
        max_len = src.shape[1]
        src_mask = torch.ones(1, 1, max_len, device=device)
        print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


    # execute_example(example_simple_model)
    return


@app.cell
def _(
    Batch,
    DummyOptimizer,
    DummyScheduler,
    LabelSmoothing,
    LambdaLR,
    greedy_decode,
    make_model,
    run_epoch,
    torch,
):
    # Part 3: A real-world example with Hugging Face + Multi30k
    import json
    from pathlib import Path

    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from torch.utils.data import DataLoader

    SRC_LANG = "de"
    TGT_LANG = "en"
    SPECIAL_TOKENS = ["<s>", "</s>", "<blank>", "<unk>"]

    # Keep canonical paper-like settings and a lighter local profile for Apple Silicon.
    PART3_CONFIGS = {
        "paper_default": {
            "N": 6,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "batch_size": 32,
            "num_epochs": 8,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
            "file_prefix": "multi30k_model_",
            "tokenizer_cache_dir": "tokenizers",
            "min_frequency": 2,
            "max_train_sentences": None,
            "max_valid_sentences": None,
        },
        "local_mps": {
            "N": 6,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "batch_size": 8,
            "num_epochs": 8,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
            "file_prefix": "multi30k_model_",
            "tokenizer_cache_dir": "tokenizers",
            "min_frequency": 2,
            "max_train_sentences": None,
            "max_valid_sentences": None,
        },
    }

    def get_device():
        # MPS first for Apple Silicon, then CUDA, then CPU fallback.
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_config(config_profile):
        if isinstance(config_profile, dict):
            return config_profile.copy()
        if config_profile not in PART3_CONFIGS:
            raise ValueError(
                f"Unknown config profile '{config_profile}'. Valid: {list(PART3_CONFIGS)}"
            )
        return PART3_CONFIGS[config_profile].copy()

    def load_hf_multi30k():
        # Hugging Face copy of the classic Multi30k German-English benchmark.
        dataset = load_dataset("bentrevett/multi30k")
        required_splits = {"train", "validation", "test"}
        missing_splits = required_splits.difference(dataset.keys())
        if missing_splits:
            raise ValueError(f"Missing expected Multi30k splits: {sorted(missing_splits)}")
        return dataset["train"], dataset["validation"], dataset["test"]

    def _train_wordlevel_tokenizer(text_iterator, min_frequency):
        # WordLevel + whitespace reproduces simple word tokenization in a modern stack.
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        return tokenizer

    def _tokenizer_paths(cache_dir):
        # Persist tokenizers to avoid rebuilding on every run.
        cache_path = Path(cache_dir)
        return (
            cache_path / "multi30k_de_wordlevel.json",
            cache_path / "multi30k_en_wordlevel.json",
        )

    def load_or_build_tokenizers(
        train_split,
        cache_dir="tokenizers",
        min_frequency=2,
        force_rebuild=False,
    ):
        src_path, tgt_path = _tokenizer_paths(cache_dir)
        src_path.parent.mkdir(parents=True, exist_ok=True)

        if force_rebuild or (not src_path.exists()) or (not tgt_path.exists()):
            print("Training word-level tokenizers from Multi30k train split ...")
            src_tokenizer = _train_wordlevel_tokenizer(
                (row[SRC_LANG] for row in train_split),
                min_frequency=min_frequency,
            )
            tgt_tokenizer = _train_wordlevel_tokenizer(
                (row[TGT_LANG] for row in train_split),
                min_frequency=min_frequency,
            )
            src_tokenizer.save(str(src_path))
            tgt_tokenizer.save(str(tgt_path))
        else:
            src_tokenizer = Tokenizer.from_file(str(src_path))
            tgt_tokenizer = Tokenizer.from_file(str(tgt_path))

        return src_tokenizer, tgt_tokenizer

    def _special_ids(tokenizer):
        # Centralized lookup so all downstream code uses consistent special IDs.
        token_ids = {}
        for token in SPECIAL_TOKENS:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Tokenizer is missing required special token: {token}")
            token_ids[token] = token_id
        return token_ids

    def _slice_split(split, max_rows):
        if max_rows is None:
            return split
        return split.select(range(min(max_rows, len(split))))

    def _encode_with_special_tokens(
        text,
        tokenizer,
        bos_id,
        eos_id,
        pad_id,
        max_padding,
        device,
    ):
        # Sequence format is: <s> tokens </s>, then right-pad/truncate to fixed length.
        token_ids = tokenizer.encode(text).ids
        token_ids = [bos_id] + token_ids + [eos_id]
        if len(token_ids) > max_padding:
            token_ids = token_ids[:max_padding]
            token_ids[-1] = eos_id

        encoded = torch.full((max_padding,), pad_id, dtype=torch.int64, device=device)
        encoded[: len(token_ids)] = torch.tensor(
            token_ids,
            dtype=torch.int64,
            device=device,
        )
        return encoded

    def collate_batch_hf(
        batch,
        src_tokenizer,
        tgt_tokenizer,
        src_ids,
        tgt_ids,
        device,
        max_padding=128,
    ):
        # Convert raw HF rows into fixed-size tensors compatible with Batch/masking code.
        src_batch = []
        tgt_batch = []
        for row in batch:
            src_batch.append(
                _encode_with_special_tokens(
                    row[SRC_LANG],
                    src_tokenizer,
                    src_ids["<s>"],
                    src_ids["</s>"],
                    src_ids["<blank>"],
                    max_padding,
                    device,
                )
            )
            tgt_batch.append(
                _encode_with_special_tokens(
                    row[TGT_LANG],
                    tgt_tokenizer,
                    tgt_ids["<s>"],
                    tgt_ids["</s>"],
                    tgt_ids["<blank>"],
                    max_padding,
                    device,
                )
            )

        return torch.stack(src_batch), torch.stack(tgt_batch)

    def create_dataloaders_hf(
        device,
        train_split,
        valid_split,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=32,
        max_padding=72,
        max_train_sentences=None,
        max_valid_sentences=None,
    ):
        # Optional split slicing makes smoke tests and local experimentation fast.
        train_data = _slice_split(train_split, max_train_sentences)
        valid_data = _slice_split(valid_split, max_valid_sentences)

        src_ids = _special_ids(src_tokenizer)
        tgt_ids = _special_ids(tgt_tokenizer)
        if src_ids["<blank>"] != tgt_ids["<blank>"]:
            raise ValueError("Source and target <blank> token ids must match.")

        def collate_fn(batch):
            return collate_batch_hf(
                batch,
                src_tokenizer,
                tgt_tokenizer,
                src_ids,
                tgt_ids,
                device,
                max_padding=max_padding,
            )

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return train_dataloader, valid_dataloader

    def _build_model_from_config(src_vocab_size, tgt_vocab_size, config, device):
        # Model architecture comes from existing make_model; this only wires config values.
        model = make_model(
            src_vocab_size,
            tgt_vocab_size,
            N=config["N"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            h=config["h"],
            dropout=config["dropout"],
        )
        return model.to(device)

    def train_model(
        config_profile="local_mps",
        force_rebuild_tokenizers=False,
    ):
        # End-to-end Part 3 training entrypoint.
        config = _resolve_config(config_profile)
        device = get_device()
        print(f"Using device: {device}")

        train_split, valid_split, _ = load_hf_multi30k()
        src_tokenizer, tgt_tokenizer = load_or_build_tokenizers(
            train_split,
            cache_dir=config["tokenizer_cache_dir"],
            min_frequency=config["min_frequency"],
            force_rebuild=force_rebuild_tokenizers,
        )

        train_dataloader, valid_dataloader = create_dataloaders_hf(
            device=device,
            train_split=train_split,
            valid_split=valid_split,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=config["batch_size"],
            max_padding=config["max_padding"],
            max_train_sentences=config["max_train_sentences"],
            max_valid_sentences=config["max_valid_sentences"],
        )

        src_vocab_size = src_tokenizer.get_vocab_size()
        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        model = _build_model_from_config(src_vocab_size, tgt_vocab_size, config, device)

        pad_idx = tgt_tokenizer.token_to_id("<blank>")
        criterion = LabelSmoothing(
            size=tgt_vocab_size,
            padding_idx=pad_idx,
            smoothing=config["label_smoothing"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["base_lr"],
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        # Same schedule form used in the original tutorial/paper discussion.
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step,
                model_size=config["d_model"],
                factor=1.0,
                warmup=config["warmup"],
            ),
        )

        class LocalTrainState:
            step = 0
            accum_step = 0
            samples = 0
            tokens = 0

        train_state = LocalTrainState()

        for epoch in range(config["num_epochs"]):
            print(f"Epoch {epoch} training")
            model.train()
            _, train_state = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train+log",
                accum_iter=config["accum_iter"],
                train_state=train_state,
            )

            epoch_checkpoint = f"{config['file_prefix']}{epoch:02d}.pt"
            torch.save(model.state_dict(), epoch_checkpoint)

            print(f"Epoch {epoch} validation")
            model.eval()
            valid_loss, _ = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )
            print(f"Validation loss: {float(valid_loss):.4f}")

        final_checkpoint = f"{config['file_prefix']}final.pt"
        torch.save(model.state_dict(), final_checkpoint)
        with open(f"{config['file_prefix']}config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        return model, src_tokenizer, tgt_tokenizer, train_dataloader, valid_dataloader, config

    def load_trained_model(
        config_profile="local_mps",
        force_retrain=False,
        force_rebuild_tokenizers=False,
    ):
        # Reuse existing checkpoint unless retraining is explicitly requested.
        config = _resolve_config(config_profile)
        device = get_device()

        train_split, valid_split, _ = load_hf_multi30k()
        src_tokenizer, tgt_tokenizer = load_or_build_tokenizers(
            train_split,
            cache_dir=config["tokenizer_cache_dir"],
            min_frequency=config["min_frequency"],
            force_rebuild=force_rebuild_tokenizers,
        )

        final_checkpoint = Path(f"{config['file_prefix']}final.pt")
        if force_retrain or not final_checkpoint.exists():
            train_model(
                config_profile=config,
                force_rebuild_tokenizers=force_rebuild_tokenizers,
            )

        model = _build_model_from_config(
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            config=config,
            device=device,
        )
        model.load_state_dict(torch.load(final_checkpoint, map_location=device))
        model.eval()

        _, valid_dataloader = create_dataloaders_hf(
            device=device,
            train_split=train_split,
            valid_split=valid_split,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=1,
            max_padding=config["max_padding"],
            max_train_sentences=config["max_train_sentences"],
            max_valid_sentences=config["max_valid_sentences"],
        )
        return model, src_tokenizer, tgt_tokenizer, valid_dataloader, config, device

    def _tokens_from_ids(tokenizer, token_ids, pad_id, eos_token="</s>"):
        # Convert token IDs back to readable text tokens for quick qualitative checks.
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == pad_id:
                continue
            token = tokenizer.id_to_token(token_id)
            if token is None:
                token = "<unk>"
            tokens.append(token)
            if token == eos_token:
                break
        return tokens

    def check_outputs(
        valid_dataloader,
        model,
        src_tokenizer,
        tgt_tokenizer,
        n_examples=5,
    ):
        # Greedy decode samples from validation to inspect translation quality.
        src_pad_id = src_tokenizer.token_to_id("<blank>")
        tgt_pad_id = tgt_tokenizer.token_to_id("<blank>")
        if src_pad_id != tgt_pad_id:
            raise ValueError("Source and target <blank> token ids must match.")
        start_symbol = tgt_tokenizer.token_to_id("<s>")

        results = []
        valid_iter = iter(valid_dataloader)
        for idx in range(n_examples):
            try:
                batch = next(valid_iter)
            except StopIteration:
                break

            rb = Batch(batch[0], batch[1], tgt_pad_id)
            model_out = greedy_decode(
                model,
                rb.src,
                rb.src_mask,
                max_len=rb.src.size(1),
                start_symbol=start_symbol,
            )[0]

            src_tokens = _tokens_from_ids(
                src_tokenizer,
                rb.src[0].tolist(),
                src_pad_id,
            )
            tgt_tokens = _tokens_from_ids(
                tgt_tokenizer,
                batch[1][0].tolist(),
                tgt_pad_id,
            )
            model_tokens = _tokens_from_ids(
                tgt_tokenizer,
                model_out.tolist(),
                tgt_pad_id,
            )

            print(f"\nExample {idx} ========\n")
            print("Source Text (Input)        :", " ".join(src_tokens))
            print("Target Text (Ground Truth) :", " ".join(tgt_tokens))
            print("Model Output               :", " ".join(model_tokens))
            results.append(
                {
                    "source_tokens": src_tokens,
                    "target_tokens": tgt_tokens,
                    "model_tokens": model_tokens,
                }
            )

        return results

    def run_model_example(
        n_examples=5,
        config_profile="local_mps",
        force_retrain=False,
        force_rebuild_tokenizers=False,
    ):
        model, src_tokenizer, tgt_tokenizer, valid_dataloader, config, device = (
            load_trained_model(
                config_profile=config_profile,
                force_retrain=force_retrain,
                force_rebuild_tokenizers=force_rebuild_tokenizers,
            )
        )
        print(f"Loaded profile: {config_profile}")
        print(f"Device: {device}")
        return check_outputs(
            valid_dataloader=valid_dataloader,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            n_examples=n_examples,
        )

    def run_part3_data_checks(config_profile="local_mps"):
        # Lightweight validations: dataset shape, tokenizer IDs, batch/mask integrity.
        config = _resolve_config(config_profile)
        train_split, valid_split, test_split = load_hf_multi30k()
        assert len(train_split) > 0 and len(valid_split) > 0 and len(test_split) > 0

        src_tokenizer, tgt_tokenizer = load_or_build_tokenizers(
            train_split,
            cache_dir=config["tokenizer_cache_dir"],
            min_frequency=config["min_frequency"],
            force_rebuild=False,
        )
        src_ids = _special_ids(src_tokenizer)
        tgt_ids = _special_ids(tgt_tokenizer)

        sample = train_split[0]
        src_encoded = src_tokenizer.encode(sample[SRC_LANG]).ids
        tgt_encoded = tgt_tokenizer.encode(sample[TGT_LANG]).ids
        assert len(src_encoded) > 0 and len(tgt_encoded) > 0

        device = get_device()
        train_dataloader, _ = create_dataloaders_hf(
            device=device,
            train_split=train_split,
            valid_split=valid_split,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=2,
            max_padding=config["max_padding"],
            max_train_sentences=8,
            max_valid_sentences=4,
        )
        batch = next(iter(train_dataloader))
        assert batch[0].shape == batch[1].shape
        rb = Batch(batch[0], batch[1], tgt_ids["<blank>"])
        assert rb.src_mask.size(0) == batch[0].size(0)
        assert rb.target_mask.size(0) == batch[1].size(0)

        return {
            "train_size": len(train_split),
            "valid_size": len(valid_split),
            "test_size": len(test_split),
            "src_vocab_size": src_tokenizer.get_vocab_size(),
            "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
            "src_special_ids": src_ids,
            "tgt_special_ids": tgt_ids,
            "batch_shape": (tuple(batch[0].shape), tuple(batch[1].shape)),
        }

    def run_part3_training_smoke():
        # One short run to verify the training path without paying full epoch cost.
        smoke_config = _resolve_config("local_mps")
        smoke_config["num_epochs"] = 1
        smoke_config["batch_size"] = 4
        smoke_config["max_train_sentences"] = 128
        smoke_config["max_valid_sentences"] = 64
        smoke_config["file_prefix"] = "multi30k_smoke_"
        train_model(config_profile=smoke_config, force_rebuild_tokenizers=False)
        checkpoint_path = Path(f"{smoke_config['file_prefix']}final.pt")
        if not checkpoint_path.exists():
            raise RuntimeError("Smoke training did not produce final checkpoint.")
        return str(checkpoint_path)

    return (train_model,)


@app.cell
def _(train_model):
    train_model("local_mps")                    # full training loop
    # or: train_model("paper_default")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
