# remus
Remote Unified Search - a high-performance GPU-native hybrid keyword/semantic search and retrieval engine for finding the needle in the haystack.

## Developer Quickstart

Create `uv` Virtual Environment and Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv sync --all-groups
```

## Testing
```bash
pytest -v -s tests/
```

## Hardware

|         System        |    CPU   |    Memory   |        GPU             |  Network  |
| :-------------------- | :------- | :---------- | :--------------------- | :-------- |
| Bare Metal (2 nodes)  | 2 x 16   | 2 x 64 GiB  | 2 x 2 RTX 3090 (24 GiB)| 64 gbps   |
| AWS `g6.12xlarge`     | 48       | 192 GiB     | 4 x L4 (24 GiB)        | 40 gbps   |
