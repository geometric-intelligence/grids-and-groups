"""Train a quadratic RNN on local or global SGC trajectories.

This file is deliberately self-contained so the pilot does not change the
project's dataset, training, or model modules. It reuses the existing model and
the local-motion support from the constructed discrete-SE(2) analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import template
from src.experiments.discrete_se2 import _LOCAL_ROTATIONS, _LOCAL_TRANSLATIONS
from src.groups.opposite import as_action_group
from src.groups.znxzn_cm import DiscreteSE2Group
from trained_networks.model import QuadraticRNN


@dataclass(frozen=True)
class ExperimentConfig:
    train_distribution: str
    seed: int
    n: int
    m: int
    sequence_length: int
    hidden_dim: int
    batch_size: int
    steps: int
    learning_rate: float
    init_scale: float
    eval_size: int
    eval_batch_size: int
    eval_interval: int
    grad_clip: float
    device: str


def local_egocentric_elements(group: DiscreteSE2Group) -> list[int]:
    """Use exactly the support defined by the constructed-network analysis."""
    return sorted(
        {
            group.encode(dx, dy, rotation)
            for dx, dy in _LOCAL_TRANSLATIONS
            for rotation in _LOCAL_ROTATIONS
        }
    )


def make_codebook(group: DiscreteSE2Group) -> tuple[np.ndarray, np.ndarray]:
    """Return the template orbit and the original-group Cayley table."""
    powers = [1.0] * len(group.irreps())
    template_vector = template.custom_fourier(group, powers)
    rms = float(np.sqrt(np.mean(template_vector**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("template must have positive finite RMS")
    template_vector = template_vector / rms

    action_group = as_action_group(group, "right")
    codebook = np.stack(
        [action_group.left_action(g, template_vector) for g in group.elements()]
    ).astype(np.float32)
    cayley = np.asarray(
        [[group.compose(g, h) for h in group.elements()] for g in group.elements()],
        dtype=np.int64,
    )
    return codebook, cayley


def sample_sequences(
    *,
    generator: torch.Generator,
    batch_size: int,
    sequence_length: int,
    group_size: int,
    distribution: str,
    local_elements: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample a global initial state followed by local or global increments."""
    sequence = torch.empty(
        (batch_size, sequence_length), dtype=torch.long, device=device
    )
    sequence[:, 0] = torch.randint(
        group_size, (batch_size,), generator=generator, device=device
    )
    if distribution == "global":
        sequence[:, 1:] = torch.randint(
            group_size,
            (batch_size, sequence_length - 1),
            generator=generator,
            device=device,
        )
    elif distribution == "local":
        local_indices = torch.randint(
            len(local_elements),
            (batch_size, sequence_length - 1),
            generator=generator,
            device=device,
        )
        sequence[:, 1:] = local_elements[local_indices]
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return sequence


def encode_sequences(
    sequence: torch.Tensor,
    codebook: torch.Tensor,
    cayley: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode inputs and all prefix-product targets after the initial state."""
    inputs = codebook[sequence]
    cumulative = sequence[:, 0]
    target_indices = []
    for step in range(1, sequence.shape[1]):
        cumulative = cayley[cumulative, sequence[:, step]]
        target_indices.append(cumulative)
    target_indices_tensor = torch.stack(target_indices, dim=1)
    targets = codebook[target_indices_tensor]
    return inputs, targets, target_indices_tensor


def make_fixed_eval_sequences(
    *,
    seed: int,
    eval_size: int,
    sequence_length: int,
    group_size: int,
    distribution: str,
    local_elements: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return sample_sequences(
        generator=generator,
        batch_size=eval_size,
        sequence_length=sequence_length,
        group_size=group_size,
        distribution=distribution,
        local_elements=local_elements,
        device=device,
    )


@torch.no_grad()
def evaluate(
    network: nn.Module,
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    cayley: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    network.eval()
    squared_error = 0.0
    num_values = 0
    num_correct = 0
    num_outputs = 0
    for start in range(0, len(sequences), batch_size):
        batch_sequence = sequences[start : start + batch_size]
        inputs, targets, target_indices = encode_sequences(batch_sequence, codebook, cayley)
        predictions = network(inputs)
        squared_error += torch.sum((predictions - targets) ** 2).item()
        num_values += targets.numel()

        flat_predictions = predictions.reshape(-1, predictions.shape[-1])
        decoded = torch.argmax(flat_predictions @ codebook.T, dim=1)
        flat_targets = target_indices.reshape(-1)
        num_correct += torch.sum(decoded == flat_targets).item()
        num_outputs += flat_targets.numel()
    return squared_error / num_values, num_correct / num_outputs


def write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "step",
        "elapsed_seconds",
        "train_batch_mse",
        "local_mse",
        "global_mse",
        "local_accuracy",
        "global_accuracy",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_reference_convention() -> None:
    """Check the fast target construction against regular-representation products."""
    group = DiscreteSE2Group(n=2, m=6)
    codebook, cayley = make_codebook(group)
    action_group = as_action_group(group, "right")
    regular = action_group.regular_rep()
    rng = np.random.default_rng(123)
    for _ in range(12):
        sequence = rng.integers(0, group.order, size=4)
        product = int(sequence[0])
        cumulative_representation = regular[int(sequence[0])]
        for element in sequence[1:]:
            product = int(cayley[product, int(element)])
            cumulative_representation = regular[int(element)] @ cumulative_representation
        reference = cumulative_representation @ codebook[0]
        np.testing.assert_allclose(reference, codebook[product], atol=1e-5)

    expected_local = {
        group.encode(dx, dy, rotation)
        for dx, dy in _LOCAL_TRANSLATIONS
        for rotation in _LOCAL_ROTATIONS
    }
    assert set(local_egocentric_elements(group)) == expected_local


def run(config: ExperimentConfig, output_dir: Path) -> None:
    if config.m != 6:
        raise ValueError("this pilot intentionally uses the existing C6 local-motion definition")
    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    group = DiscreteSE2Group(n=config.n, m=config.m)
    codebook_np, cayley_np = make_codebook(group)
    codebook = torch.tensor(codebook_np, device=device)
    cayley = torch.tensor(cayley_np, dtype=torch.long, device=device)
    local_elements_list = local_egocentric_elements(group)
    local_elements = torch.tensor(local_elements_list, dtype=torch.long, device=device)

    metadata = asdict(config)
    metadata.update(
        {
            "group_order": group.order,
            "local_support_size": len(local_elements_list),
            "local_elements": local_elements_list,
            "local_elements_decoded": [group.decode(g) for g in local_elements_list],
            "supervision": "all_prefixes",
            "action_side": "right",
        }
    )
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    network = QuadraticRNN(
        group_size=group.order,
        hidden_dim=config.hidden_dim,
        k=config.sequence_length,
        init_scale=config.init_scale,
        return_all_outputs=True,
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(config.seed + 10_000)
    local_eval = make_fixed_eval_sequences(
        seed=100_000 + config.seed,
        eval_size=config.eval_size,
        sequence_length=config.sequence_length,
        group_size=group.order,
        distribution="local",
        local_elements=local_elements,
        device=device,
    )
    global_eval = make_fixed_eval_sequences(
        seed=200_000 + config.seed,
        eval_size=config.eval_size,
        sequence_length=config.sequence_length,
        group_size=group.order,
        distribution="global",
        local_elements=local_elements,
        device=device,
    )

    rows: list[dict[str, float | int]] = []
    start_time = time.time()
    train_batch_mse = float("nan")
    evaluation_steps = set(range(0, config.steps + 1, config.eval_interval)) | {config.steps}

    for step in range(config.steps + 1):
        if step in evaluation_steps:
            local_mse, local_accuracy = evaluate(
                network, local_eval, codebook, cayley, config.eval_batch_size
            )
            global_mse, global_accuracy = evaluate(
                network, global_eval, codebook, cayley, config.eval_batch_size
            )
            row = {
                "step": step,
                "elapsed_seconds": time.time() - start_time,
                "train_batch_mse": train_batch_mse,
                "local_mse": local_mse,
                "global_mse": global_mse,
                "local_accuracy": local_accuracy,
                "global_accuracy": global_accuracy,
            }
            rows.append(row)
            write_metrics(output_dir / "metrics.csv", rows)
            print(
                f"step={step:6d} train={train_batch_mse:.6g} "
                f"local={local_mse:.6g}/{local_accuracy:.3f} "
                f"global={global_mse:.6g}/{global_accuracy:.3f}",
                flush=True,
            )
        if step == config.steps:
            break

        network.train()
        sequence = sample_sequences(
            generator=train_generator,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            group_size=group.order,
            distribution=config.train_distribution,
            local_elements=local_elements,
            device=device,
        )
        inputs, targets, _ = encode_sequences(sequence, codebook, cayley)
        optimizer.zero_grad(set_to_none=True)
        predictions = network(inputs)
        loss = torch.mean((predictions - targets) ** 2)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
        optimizer.step()
        train_batch_mse = float(loss.detach())

    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": metadata,
            "metrics": rows,
        },
        output_dir / "final_model.pt",
    )
    (output_dir / "COMPLETE").write_text("complete\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-distribution", choices=("local", "global"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--init-scale", type=float, default=5e-2)
    parser.add_argument("--eval-size", type=int, default=8_192)
    parser.add_argument("--eval-batch-size", type=int, default=1_024)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_reference_convention()
    if args.validate_only:
        print("validation passed")
        return
    config = ExperimentConfig(
        train_distribution=args.train_distribution,
        seed=args.seed,
        n=args.n,
        m=args.m,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        init_scale=args.init_scale,
        eval_size=args.eval_size,
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
