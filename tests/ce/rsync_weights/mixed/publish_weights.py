import argparse
import time

from checkpoint_transfer import CheckpointTransfer
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Paddle checkpoint shards for FastDeploy rsync update tests")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backend", default="mooncake")
    parser.add_argument("--bucket-size-mb", type=int, default=2048)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--device", default=None)
    parser.add_argument("--global-rank", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=600.0)
    return parser.parse_args()


def load_state_dict_cpu(path):
    state_dict = {}
    with safe_open(path, framework="paddle", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def main():
    args = parse_args()

    # state_dict = paddle.load(args.state_path, safetensors=True)
    state_dict = load_state_dict_cpu(args.state_path)
    ct = CheckpointTransfer(
        backend=args.backend,
        bucket_size_mb=args.bucket_size_mb,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        global_rank=args.global_rank,
        group_size=args.group_size,
        device=args.device,
    )
    ct.initialize()
    stats = ct.send(state_dict, step_id=args.version, verify_checksum=args.verify_checksum)
    ct.mark_step_ready(args.version)

    if args.ready_file:
        with open(args.ready_file, "w", encoding="utf-8") as f:
            f.write(f"{args.version}\n")

    print(
        {
            "version": args.version,
            "state_path": args.state_path,
            "global_rank": args.global_rank,
            "group_size": args.group_size,
            "num_tensors": len(state_dict),
            "send_stats": stats,
        }
    )

    if args.hold_seconds < 0:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    elif args.hold_seconds > 0:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
