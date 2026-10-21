# Model and dependency notices

The model is [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), pinned at revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. The upstream model card declares Apache-2.0, 384-dimensional embeddings, attention-mask mean pooling, and a 128-token limit. This implementation follows those pooling and length settings. Model weights are downloaded separately and are not redistributed in this repository.

The public model API resolved that revision during implementation. Downloads select only configuration, tokenizer assets, and safetensors weights. Execution uses `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`. The download command explicitly uses the network; inference does not need it.

The pinned runtime in `requirements-local.lock` is the measured Intel macOS reference environment, not a universal platform lock or security audit. PyTorch 2.2.2 is used for that platform. Users on another platform must record a compatible stack and rerun the experiment; portable CI deliberately excludes neural libraries. Python and third-party packages retain their respective licenses.
