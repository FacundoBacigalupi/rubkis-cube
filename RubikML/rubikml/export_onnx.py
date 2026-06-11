"""
Policy-only export (for greedy solver):
  python -m rubikml.export_onnx --checkpoint checkpoints/best.pt

Policy+value export (for beam search — much better on deep scrambles):
  python -m rubikml.export_onnx --checkpoint checkpoints/best.pt --include-value
"""
import argparse
import os

import torch

from .solve import load_model


def _check_onnx(path: str):
    try:
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m)
        print("ONNX model check: OK")
    except ImportError:
        print("onnx package not installed — skipping model check")


def export_onnx(checkpoint: str, output: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    device = torch.device("cpu")
    model = load_model(checkpoint, device)
    model.eval()

    dummy = torch.zeros(1, 324)

    class PolicyOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return out[0] if isinstance(out, tuple) else out

    torch.onnx.export(
        PolicyOnly(model),
        dummy,
        output,
        input_names=["cube_state"],
        output_names=["move_logits"],
        dynamic_axes={"cube_state": {0: "batch"}, "move_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported policy-only → {output}")
    print("  Input  : cube_state  float32[batch, 324]")
    print("  Output : move_logits float32[batch, 12]")
    _check_onnx(output)


def export_onnx_pv(checkpoint: str, output: str):
    """Export policy + value heads so beam search can use distance estimates."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    device = torch.device("cpu")
    model = load_model(checkpoint, device)
    model.eval()

    dummy = torch.zeros(1, 324)
    test_out = model(dummy)
    if not isinstance(test_out, tuple):
        raise RuntimeError(
            "Checkpoint has no value head (model_type='policy'). "
            "Train a policy_value model first."
        )

    class PolicyValue(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            logits, value = self.inner(x)
            return logits, value

    torch.onnx.export(
        PolicyValue(model),
        dummy,
        output,
        input_names=["cube_state"],
        output_names=["move_logits", "value"],
        dynamic_axes={
            "cube_state":  {0: "batch"},
            "move_logits": {0: "batch"},
            "value":       {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported policy+value → {output}")
    print("  Input   : cube_state  float32[batch, 324]")
    print("  Output0 : move_logits float32[batch, 12]")
    print("  Output1 : value       float32[batch,  1]")
    _check_onnx(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default=None,
                        help="Output path (default: exports/rubik_policy.onnx or "
                             "exports/rubik_pv.onnx when --include-value is set)")
    parser.add_argument("--include-value", action="store_true",
                        help="Export policy + value heads for value-guided beam search")
    args = parser.parse_args()

    if args.include_value:
        out = args.output or "exports/rubik_pv.onnx"
        export_onnx_pv(args.checkpoint, out)
        print("\nNext steps:")
        print(f"  copy {out} to Assets/ML/rubik_pv.onnx")
        print("  assign it to NeuralSolver → Pv Model Asset in the Unity Inspector")
    else:
        out = args.output or "exports/rubik_policy.onnx"
        export_onnx(args.checkpoint, out)


if __name__ == "__main__":
    main()
