"""Facts about the recorded data that bound how far the results extend: person size and missing joints.

    python -m experiments.dataset_stats
"""
import numpy as np

from uavsignals.features import fill_missing

from .common import load, parser, save_json


def main():
    args = parser(__doc__).parse_args()
    videos = load(args)
    heights, missing = [], []
    for v in videos.values():
        x = fill_missing(v.keypoints, v.visible)
        head_to_ankle = np.max(x[:, [10, 13], 1], axis=1) - x[:, 0, 1]     # nose to the lower ankle, pixels
        heights.append(float(np.median(head_to_ankle)))
        missing.append(float(1 - v.visible.mean()))
    h = np.array(heights)
    result = dict(videos=len(videos), image='1920 x 1080',
                  person_height_px=dict(median=float(np.median(h)), p5=float(np.percentile(h, 5)),
                                        p95=float(np.percentile(h, 95)), min=float(h.min()), max=float(h.max())),
                  person_height_fraction_of_frame=float(np.median(h) / 1080),
                  missing_joints=dict(mean=float(np.mean(missing)), max=float(np.max(missing))))
    print(f'Person height (nose to ankle): median {result["person_height_px"]["median"]:.0f} px, '
          f'5-95% {result["person_height_px"]["p5"]:.0f}-{result["person_height_px"]["p95"]:.0f} px '
          f'({100 * result["person_height_fraction_of_frame"]:.0f}% of frame height)')
    print(f'Missing joints: {100 * result["missing_joints"]["mean"]:.1f}% on average, '
          f'{100 * result["missing_joints"]["max"]:.1f}% at most')
    save_json(result, 'dataset_stats.json')


if __name__ == '__main__':
    main()
