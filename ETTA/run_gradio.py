# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# modified from stable-audio-tools under the MIT license

from stable_audio_tools.interface.gradio import create_ui

import torch

def main(args):
    torch.manual_seed(42)

    interface = create_ui(
        ckpt_path=args.ckpt_path,
        model_half=args.model_half
    )
    interface.queue()
    interface.launch(server_name="0.0.0.0", server_port=7680, share=args.share, auth=(args.username, args.password) if args.username is not None else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    args = parser.parse_args()
    main(args)