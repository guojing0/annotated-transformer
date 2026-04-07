import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # Some imports

    import os
    from os.path import exists
    import torch
    import torch.nn as nn
    from torch.nn.functional import log_softmax, pad
    import math
    import copy
    import time
    from torch.optim.lr_scheduler import LambdaLR
    import pandas as pd
    import altair as alt
    from torch.utils.data import DataLoader
    from torchtext.vocab import build_vocab_from_iterator
    import torchtext.datasets as datasets
    import spacy
    import warnings
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        from torchtext.data.functional import to_map_style_dataset
    except ImportError:
        # torchtext versions without this helper can still use an equivalent
        # local map-style wrapper for iterable datasets.
        from torch.utils.data import Dataset

        class _MapStyleDataset(Dataset):
            def __init__(self, iterator):
                self._data = list(iterator)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        def to_map_style_dataset(iterator):
            return _MapStyleDataset(iterator)
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


    return DummyOptimizer, DummyScheduler, execute_example, show_example


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
            self.layers = clones(layer, 6)
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
            nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
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
def _(memory, subsequent_mask, torch):
    # Greedy decoding

    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        memeory = model.encode(src, src_mask)
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
    execute_example,
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


    execute_example(example_simple_model)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
