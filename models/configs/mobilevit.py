from typing import Dict

def get_model_config(args) -> Dict:
    mode = args.mode
    head_dim, num_heads = args.head_dim, args.head_num

    if head_dim is not None:
        if num_heads is not None:
            print(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )
    elif num_heads is not None:
        if head_dim is not None:
            print(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )

    if mode == "xxs":
        config = {
            "expansion": 2,
            "dims": [64, 80, 96],
            "channels": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        }
    elif mode == "xs":
        config = {
            "expansion": 1,
            "dims": [96, 120, 144],
            "channels": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        }
    elif mode == "s":
        config = {
            "expansion": 1,
            "dims": [144, 192, 240],
            "channels": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        }
    else:
        raise NotImplementedError

    return config
