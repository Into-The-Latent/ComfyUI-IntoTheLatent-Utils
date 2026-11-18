"""One CUDA stream for every graph capture, kept for the life of the process.

The graphs here used to be captured on several streams. Some warmed up on a stream of their own that
was dropped after the capture while ``torch.cuda.graph`` recorded them on PyTorch's own default capture
stream, others (prefill, codec, text encoder) recorded on a stream of their own. PyTorch's native
allocator does not mind. Under the cudaMallocAsync allocator, which ties memory to streams and is what
ComfyUI switches on by default, the replay of a graph is then an illegal memory access as soon as a
second graph exists: the depth decoder's two buckets alone are enough. Warmed up and captured on one
stream that is never dropped, the same graphs replay correctly under both allocators.

That one stream is PyTorch's default capture stream, so a capture that names no stream lands on it too.
"""
from __future__ import annotations

import torch

_STREAMS: dict[int, torch.cuda.Stream] = {}


def capture_stream(device) -> torch.cuda.Stream:
    """The stream every graph on this device is warmed up and captured on."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        default = torch.cuda.graph.default_capture_stream
        if default is None and index == torch.cuda.current_device():
            # What torch.cuda.graph would make on its first use without a stream.
            default = torch.cuda.graph.default_capture_stream = torch.cuda.Stream()
        if default is not None and default.device.index == index:
            stream = default
        else:
            stream = torch.cuda.Stream(device=device)
        _STREAMS[index] = stream
    return stream
