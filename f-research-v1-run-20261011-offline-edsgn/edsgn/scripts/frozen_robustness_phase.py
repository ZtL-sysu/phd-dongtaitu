"""CLI for one complete, already-frozen robustness test intervention."""
import argparse
import json
from pathlib import Path
from edsgn.robustness import evaluate_frozen_robustness
from edsgn.scheduling import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8-sig"))
    policy = Policy(**json.loads(Path(args.policy_config).read_text(encoding="utf-8-sig"))) if args.policy_config else Policy()
    evaluate_frozen_robustness(args.checkpoint, "test", args.output, recipe, policy=policy, device_name=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
