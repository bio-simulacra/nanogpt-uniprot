# NanoGPT Uniprot

This is a simple implementation of NanoGPT for Uniprot.

This is largely borrowed from [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/tree/master), and we thank him for sharing his code.


# Highlights

- We use `uv` for package management. To install the development version, run:

```
uv pip install -e .
```

# Quickstart

To execute the training, run the following commands.

```
uv run src/nanogpt_uniprot/data/cached_fineweb10B.py
bash run.sh
```

## Prerequesites
### On WSL
install GCC
```
sudo apt update
sudo apt install build-essential
```
After installation, you can verify that GCC is installed by running:
```
gcc --version
```


# TODOs

- [x] Reproduce the original modded-nanogpt demo
- [ ] Set up config to be Hydra-compatible
- [ ] Reproduce the original modded-nanogpt with UniRef data
  - [ ] Download scripts for UniRef 100 and 50
    - [ ] Splits for train/val/test
  - [ ] Create dataloaders for uniref
    - [ ] Pre-tokenized sequences
  - [ ] Performance evaluation
- [ ] Get running demo
