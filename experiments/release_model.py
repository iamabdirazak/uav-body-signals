"""Train the released model on all 119 videos (every actor) and save it to weights/.

This is the model used by the live demo and by Recogniser.from_pretrained(). Its accuracy cannot
be measured on UAV-GESTURE, since no actor is held out; the fold results describe the method.

    python -m experiments.release_model
"""
import os

import torch

from uavsignals.recogniser import DEFAULT_WEIGHTS
from uavsignals.training import CONFIGS, train

from .common import load, parser


def main():
    p = parser(__doc__)
    p.add_argument('--out', default=DEFAULT_WEIGHTS)
    p.add_argument('--epochs', type=int, default=35)
    args = p.parse_args()
    videos = load(args)
    net = train(CONFIGS['lstm'], videos, sorted(videos), epochs=args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
