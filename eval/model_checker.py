

from typing import Optional


def compute_shard_invariant_weight_hash(
    model_repo: str, revision: str = None, sample_bytes: int = 2048,
) -> Optional[str]:
    """
    Compute a shard-invariant hash from actual weight CONTENT, not just metadata.

    For each safetensors file, reads the header to locate tensors, then reads
    the first `sample_bytes` of each tensor's raw data.  Tensors are sorted by
    name so the hash is identical regardless of how the model is sharded.

    Two models with the same architecture but different trained weights will
    produce DIFFERENT hashes (unlike compute_tensor_metadata_hash which only
    checks names/shapes/dtypes).

    Returns hex digest or None on failure.
    """
    import struct
    try:
        info = model_info(model_repo, revision=revision, files_metadata=True)
        st_files = sorted(
            [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".safetensors")]
        )
        if not st_files:
            logger.warning(f"No safetensors files in {model_repo}")
            return None

        # Collect (tensor_name, metadata_str, weight_sample_bytes) across all shards
        tensor_samples = {}  # name -> (meta_str, weight_bytes)
        for fname in st_files:
            file_path = hf_hub_download(
                repo_id=model_repo, filename=fname, revision=revision,
            )
            with open(file_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = json.loads(f.read(header_size))
                data_offset = 8 + header_size  # where tensor data starts in file

                for tensor_name, tensor_info in header_json.items():
                    if tensor_name == "__metadata__":
                        continue
                    dtype = tensor_info.get("dtype", "")
                    shape = tuple(tensor_info.get("shape", []))
                    offsets = tensor_info.get("data_offsets", [0, 0])
                    begin, end = offsets[0], offsets[1]
                    tensor_size = end - begin
                    read_size = min(sample_bytes, tensor_size)

                    # Seek to the tensor data and read a sample
                    f.seek(data_offset + begin)
                    weight_sample = f.read(read_size)

                    meta_str = f"{tensor_name}:{shape}:{dtype}"
                    tensor_samples[tensor_name] = (meta_str, weight_sample)

        # Sort by tensor name for deterministic ordering
        hasher = hashlib.sha256()
        for name in sorted(tensor_samples.keys()):
            meta_str, weight_sample = tensor_samples[name]
            hasher.update(meta_str.encode())
            hasher.update(weight_sample)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Shard-invariant weight hash failed for {model_repo}: {e}")
        return None
