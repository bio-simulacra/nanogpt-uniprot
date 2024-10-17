# NanoGPT Uniprot

This is a simple implementation of NanoGPT for Uniprot.

This is largely borrowed from [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/tree/master), and we thank him for releasing his code.


# Highlights

- We use `uv` for package management. To install the development version, run:

```
uv pip install -e .
```

# Quickstart

To execute the training, run the following commands.

```
uv run src/nanogpt_uniprot/data/cached_fineweb10B.py
uv run src/nanogpt_uniprot/train_gpt2.py
```

# TODOs

- [ ] Reproduce the original modded-nanogpt demo
- [ ] Reproduce the original modded-nanogpt with UniRef data
  - [ ] Download scripts for UniRef 100 and 50
  - [ ] Create dataloaders for uniref
    - [ ] Pre-tokenized sequences
  - [ ] Get running demo
