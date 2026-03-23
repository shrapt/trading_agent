"""
XAUUSD Trading Agent – Main entry point
========================================

Commands
--------
  python main.py train                          # train from scratch
  python main.py train --episodes 200 --resume  # resume training
  python main.py eval                           # evaluate best model
  python main.py eval --verbose                 # step-by-step output
  python main.py eval --full                    # evaluate on full dataset
  python main.py demo                           # short demo run (50 episodes)
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_train(args):
    from src.train import train
    train(num_episodes=args.episodes, resume=args.resume)


def cmd_eval(args):
    from src.evaluate import evaluate_model
    from src import config
    model = args.model or config.BEST_MODEL_PATH
    evaluate_model(model_path=model,
                   use_test_split=not args.full,
                   verbose=args.verbose)


def cmd_demo(args):
    """Quick 50-episode demo so you can see it learning in real time."""
    from src.train import train
    train(num_episodes=50, resume=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAUUSD Self-Learning Trading Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the agent")
    p_train.add_argument("--episodes", type=int, default=500)
    p_train.add_argument("--resume", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate the trained agent")
    p_eval.add_argument("--model",   default=None)
    p_eval.add_argument("--full",    action="store_true")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # demo
    p_demo = sub.add_parser("demo", help="Quick 50-episode demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)
