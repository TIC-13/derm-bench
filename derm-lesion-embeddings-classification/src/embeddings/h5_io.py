from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_DEFAULT_CHUNK = 4096


def h5_path_from_stem(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return p.with_suffix(".h5")
    if p.suffix.lower() != ".h5":
        return p.with_suffix(".h5")
    return p


def _decode_strings(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind in ("S", "O", "U"):
        return np.array([s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in out])
    return out.astype(str)

def _label_dataset_name(h5_file: h5py.File) -> str:
    if "benign_malignant" in h5_file:
        return "benign_malignant"
    if "labels" in h5_file:
        return "labels"
    raise ValueError(
        f"H5 file missing label dataset ['benign_malignant' or 'labels']: "
        f"{h5_file.filename}"
    )

def _require_datasets(h5_file: h5py.File, names: List[str]) -> None:
    missing = [n for n in names if n not in h5_file]
    if missing:
        raise ValueError(f"H5 file missing datasets {missing}: {h5_file.filename}")


def read_partition_h5(path: str | Path, expand_embeddings: bool = True) -> pd.DataFrame:
    h5_path = h5_path_from_stem(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        _require_datasets(f, ["img_id"])
        label_name = _label_dataset_name(f)

        img_ids = _decode_strings(f["img_id"][:])
        labels = _decode_strings(f[label_name][:])

        data = {
            "img_id": img_ids,
            "benign_malignant": labels,
        }

        if "embeddings" in f and expand_embeddings:
            emb = f["embeddings"][:]
            for j in range(emb.shape[1]):
                data[f"embedding_{j}"] = emb[:, j]

        return pd.DataFrame(data)


def read_embeddings_matrix(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    h5_path = h5_path_from_stem(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        _require_datasets(f, ["embeddings"])
        label_name = _label_dataset_name(f)

        X = f["embeddings"][:].astype(np.float32)
        y = _decode_strings(f[label_name][:])
        return X, y

class H5PartitionWriter:

    def __init__(
        self,
        path: str | Path,
        *,
        chunksize: int = _DEFAULT_CHUNK,
        compression: str = "gzip",
    ):
        self.path = h5_path_from_stem(path)
        self.tmp_path = self.path.with_suffix(".h5.tmp")
        self.chunksize = max(1, int(chunksize))
        self.compression = compression
        self._file: Optional[h5py.File] = None
        self._img_id_ds = None
        self._label_ds = None
        self._emb_ds = None
        self._embedding_dim: Optional[int] = None
        self._total_rows = 0

    def __enter__(self) -> "H5PartitionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self._file = h5py.File(self.tmp_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.attrs["num_rows"] = self._total_rows
            if self._embedding_dim is not None:
                self._file.attrs["embedding_dim"] = self._embedding_dim
            self._file.close()
            self._file = None

        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()

    def append_row(self, img_id: str, label: str, emb: np.ndarray) -> None:
        emb = np.asarray(emb, dtype=np.float32).ravel()
        if self._embedding_dim is None:
            self._embedding_dim = emb.shape[0]
            self._init_datasets()
        elif emb.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._embedding_dim}, got {emb.shape[0]}"
            )
        self._append_batch([img_id], [label], emb.reshape(1, -1))

    def append_batch(
        self,
        img_ids: List[str],
        labels: List[str],
        embeddings: np.ndarray,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
            self._init_datasets()
        self._append_batch(img_ids, labels, embeddings)

    def _init_datasets(self) -> None:
        assert self._file is not None
        dim = self._embedding_dim
        self._img_id_ds = self._file.create_dataset(
            "img_id",
            shape=(0,),
            maxshape=(None,),
            dtype=_STRING_DTYPE,
            chunks=True,
        )
        self._label_ds = self._file.create_dataset(
            "benign_malignant",
            shape=(0,),
            maxshape=(None,),
            dtype=_STRING_DTYPE,
            chunks=True,
        )
        self._emb_ds = self._file.create_dataset(
            "embeddings",
            shape=(0, dim),
            maxshape=(None, dim),
            dtype=np.float32,
            chunks=(min(self.chunksize, 4096), dim),
            compression=self.compression,
        )

    def _append_batch(
        self,
        img_ids: List[str],
        labels: List[str],
        embeddings: np.ndarray,
    ) -> None:
        n = len(img_ids)
        if n == 0:
            return
        old_size = self._total_rows
        new_size = old_size + n

        self._img_id_ds.resize((new_size,))
        self._label_ds.resize((new_size,))
        self._emb_ds.resize((new_size, self._embedding_dim))

        self._img_id_ds[old_size:new_size] = np.array(img_ids, dtype=object)
        self._label_ds[old_size:new_size] = np.array(labels, dtype=object)
        self._emb_ds[old_size:new_size, :] = embeddings

        self._total_rows = new_size
        if self._file is not None:
            self._file.flush()


@contextmanager
def open_embeddings_h5_writer(
    path: str | Path,
    *,
    chunksize: int = _DEFAULT_CHUNK,
    compression: str = "gzip",
) -> Iterator[H5PartitionWriter]:
    with H5PartitionWriter(path, chunksize=chunksize, compression=compression) as writer:
        yield writer
