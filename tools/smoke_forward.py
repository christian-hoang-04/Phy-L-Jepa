from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectures import MaskedMuellerJEPA
from physics_features import CloudeTransformerEncoder


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny synthetic JEPA forward/backward smoke test.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--predictor-token-dim", type=int, default=16)
    parser.add_argument("--predictor-hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    torch.manual_seed(42)
    encoder = CloudeTransformerEncoder(
        patch_size=1,
        embed_dim=args.embed_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=0.1,
        image_size=args.image_size,
    )
    model = MaskedMuellerJEPA(
        encoder=encoder,
        predictor_depth=2,
        dropout=0.1,
        mask_ratio=0.5,
        ema_momentum=0.99,
        loss="smooth_l1",
        predictor_token_dim=args.predictor_token_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    ).to(device)

    x = torch.randn(args.batch_size, 16, args.image_size, args.image_size, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    out = model(x, x)
    loss = out["loss"].mean()
    loss.backward()
    optimizer.step()
    model.update_target()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    result = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_tokens": model.context_encoder.num_patches,
        "trainable_parameters": count_parameters(model),
        "loss": float(loss.detach().cpu()),
        "elapsed_sec": elapsed,
    }
    if device.type == "cuda":
        result["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
