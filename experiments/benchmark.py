"""Model size and inference time per 30-frame window, batch 1, one CPU thread.

Needs no data. Run it on the hardware you plan to deploy on.

    python -m experiments.benchmark
"""
import platform
import time

import torch

from uavsignals.constants import WINDOW
from uavsignals.training import CONFIGS
from uavsignals.models import build_model

from .common import parser, save_json


def main():
    p = parser(__doc__)
    p.add_argument('--repeats', type=int, default=2000)
    args = p.parse_args()
    torch.set_num_threads(1)
    results = dict(machine=f'{platform.system()} {platform.machine()}', processor=platform.processor())
    print(f'{results["machine"]}, 1 thread\n')
    for arch, name in [('stacked_lstm', 'stacked_lstm_body'), ('lstm', 'lstm'), ('tcn', 'tcn'), ('stgcn', 'stgcn')]:
        cfg = CONFIGS[name]
        net = build_model(arch, cfg.input_dim).eval()
        x = torch.randn(1, WINDOW, cfg.input_dim)
        with torch.no_grad():
            for _ in range(100):
                net(x)
            start = time.perf_counter()
            for _ in range(args.repeats):
                net(x)
        ms = (time.perf_counter() - start) / args.repeats * 1000
        params = sum(t.numel() for t in net.parameters())
        results[arch] = dict(params=params, ms_per_window=ms)
        print(f'{arch:13s} {params / 1000:5.0f}k params   {ms:.2f} ms per window')
    save_json(results, 'benchmark.json')


if __name__ == '__main__':
    main()
