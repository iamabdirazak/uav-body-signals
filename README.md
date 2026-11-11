# Body-Signal Recognition for Commanding Drones

A lightweight system that lets a person on the ground direct a drone with standard aircraft-handling body signals, such as *hover*, *land*, *move left* or *wave off*, seen through the drone's own camera. It works from 2D body keypoints, so it runs with any pose estimator. It is trained to hold up when the drone looks down at steep angles or tilts, and a safety gate decides when a recognised signal is allowed to become a command. This repository is the implementation behind the paper *Toward Viewpoint-Robust Body-Signal Recognition for Commanding UAVs in Search and Rescue* (IEEE USYS 2026). It contains the library, the trained model, a live webcam demo and one script for every result in the paper, so the work can be checked and built on, on ordinary hardware.

[![IEEE USYS 2026](https://img.shields.io/badge/IEEE-USYS%202026-1c1c1e?style=flat-square)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-1c1c1e?style=flat-square)](LICENSE)

<img src="assets/demo.png" alt="The live demo: guidance on the left, the tracked skeleton in the centre, the recognised signal and the safety gate's state on the right." width="100%">

<img src="assets/highlights.png" alt="91.2% video accuracy on unseen people; 86.0% at 70° camera elevation versus 38.4% unaugmented; 160 times fewer wrong actions with the safety gate; 2.8 ms per decision on one CPU thread." width="100%">

<sub>Results from offline evaluation on recorded aerial video ([UAV-GESTURE](https://asankagp.github.io/uavgesture), actor-disjoint folds). See [Limitations](#limitations).</sub>

## Quick start

You need Python 3.9 or later and no GPU. To install and check everything works:

```bash
git clone https://github.com/iamabdirazak/uav-body-signals.git
cd uav-body-signals
pip install -r requirements.txt
python -m pytest tests
```

To try the live demo with your webcam:

```bash
pip install -r demo/requirements.txt
python demo/run.py
```

Stand 2 to 4 m from the camera with your whole body in view. Perform *have command* until the right-hand panel shows **Ready**, then perform the signal shown on the left. Press N for the next signal, V to switch view, R to reset and Q to quit. The demo uses MediaPipe on a ground-level camera, while the model was trained on keypoints seen from a drone, so treat it as a demonstration rather than an evaluation.

<details>
<summary><b>Repository layout</b></summary>

```
uavsignals/           the library
  constants.py        signal names, skeleton layout, timing
  data.py             loading the UAV-GESTURE keypoints
  protocol.py         the three actor-disjoint folds
  features.py         body-centred features
  augment.py          viewpoint augmentation and test-time conditions
  models.py           proposed LSTM and the baselines (stacked LSTM, TCN, ST-GCN)
  training.py         configurations, training and evaluation
  gate.py             the safety gate
  recogniser.py       streaming recogniser: keypoints in, commands out
experiments/          one script per result in the paper
demo/                 live webcam demo
weights/              released model, trained on all actors
results/              numbers from the paper, and from a re-run on another machine
assets/               figures for this page
scripts/get_data.sh   downloads the keypoints and splits
tests/                fast checks that need no dataset
```

</details>

## How it works

<img src="assets/pipeline.png" alt="Pipeline: camera stream, pose estimation, body-centred features, temporal model, safety gate." width="100%">

1. **Pose estimation.** Any 2D estimator gives 14 body joints per frame. The paper uses the OpenPose keypoints released with UAV-GESTURE, and the demo uses MediaPipe ([`demo/pose.py`](demo/pose.py)).
2. **Body-centred features.** Each 1.2 s window is centred on the neck and scaled by torso length, so it no longer matters where the person is in the frame or how far away they are. Missing joints are interpolated and velocities are added ([`features.py`](uavsignals/features.py)).
3. **Temporal model.** A two-layer, 128-unit LSTM classifies the window into one of 13 signals. Because it is unidirectional, it can update frame by frame on board ([`models.py`](uavsignals/models.py)).
4. **Viewpoint augmentation.** During training only, each window is shown as a drone might see it: from above, tilted, with joints missing, noisy, or at a low pose rate ([`augment.py`](uavsignals/augment.py)).
5. **Safety gate.** Every 0.2 s, a signal is accepted only if it wins 6 of the last 8 windows with mean confidence of at least 0.7. *Have command* arms the gate, the next accepted signal is executed, and *wave off* always goes through ([`gate.py`](uavsignals/gate.py)).

<img src="assets/augmentation.png" alt="Viewpoint augmentation of one recorded pose: elevation 60°, roll 10°, dropout and noise." width="100%">

The 13 signals come from the NATOPS aircraft-handling manual: *all clear, have command, hover, land, landing direction, move ahead, move downward, move to left, move to right, move upward, not clear, slow down* and *wave off*.

## Use it in your own system

`Recogniser` takes one frame of keypoints at a time, from any pose estimator at any frame rate. It resamples the stream to 25 Hz, classifies every 0.2 s and applies the safety gate.

```python
from uavsignals import Recogniser

recogniser = Recogniser.from_pretrained()          # weights/uavsignals_lstm.pt

for t, keypoints, visible in pose_stream():        # seconds, (14, 2) pixels, (14,) bool
    out = recogniser.push(t, keypoints, visible)
    if out and out.command:
        send_to_flight_controller(out.command)     # e.g. 'hover', 'land', 'wave_off'
```

- **Keypoint order.** Keypoints follow the COCO body order without eyes and ears: nose, neck, then the right and left shoulder, elbow and wrist, then the right and left hip, knee and ankle. If your estimator has no neck, use the midpoint of the shoulders.
- **Speed.** The pose estimator sets the frame rate, not the classifier, which takes about 3 ms per decision on one CPU thread. Accuracy holds from 25 fps down to 3 fps. Run `python -m experiments.benchmark` on your target device.
- **Customising.** To make the gate stricter or looser, pass `k`, `n` and `tau` to `from_pretrained`. For a different pose estimator or new signals, retrain with [`training.py`](uavsignals/training.py). The released model was trained on all 119 videos (`python -m experiments.release_model`), so its accuracy can't be measured on this dataset.

## Reproduce the results

First download the UAV-GESTURE keypoints and splits (16 MB; the videos are not needed):

```bash
bash scripts/get_data.sh
```

Each result then has one command. The scripts write to `runs/`, and `python -m experiments.figures --from-runs` redraws the figures below from your own numbers. Times are for a 2-core CPU.

**Recognition on unseen people.** Run `python -m experiments.train` (about 45 min) and `python -m experiments.baselines` (about 4 min). Expressing keypoints relative to the body is what makes the task learnable from a moving camera, and augmentation brings a keypoint-only model level with an RGB and optical-flow baseline.

<img src="assets/ablation.png" alt="Video accuracy for each model, from 15.8% for image coordinates to 91.2% for the proposed model." width="100%">

**Robustness to deployment conditions.** Run `python -m experiments.robustness` (about 1 min). The conditions are applied to test keypoints without retraining. Viewpoint is the main failure mode, and augmentation removes most of it. Heavy keypoint noise remains a weakness.

<img src="assets/robustness.png" alt="Accuracy against camera elevation, roll, missing joints, keypoint noise and pose rate." width="100%">

**Safety gate on command streams.** Run `python -m experiments.safety_gate` (seconds). The gate gives up some responsiveness for far fewer wrong actions. These operating points were computed on the test streams, so read them as a trade-off rather than a tuned setting.

<img src="assets/safety_gate.png" alt="Executed commands and wrong actions per stream for four gate policies." width="100%">

<details>
<summary><b>Evaluation protocol</b></summary>

<br>

All results use three actor-disjoint folds, with test actors {S13, S14, S15}, {S1, S3, S4} and {S5, S6, S9}. Folds 2 and 3 are the dataset's official splits. Official split 1 lists actor S12 in both its training and test lists, so all S12 videos are moved to training ([`protocol.py`](uavsignals/protocol.py)). Video accuracy averages window probabilities over each clip. Each configuration is trained once per fold with seed 0, and no hyperparameters were tuned.

</details>

<details>
<summary><b>Paper and re-run compared</b></summary>

<br>

Training is seeded, but results still depend slightly on the PyTorch version and the processor. This code was re-run on a different machine from the paper's (Linux, 2 CPU cores, PyTorch 2.15):

| Result | Paper | Re-run |
|:--|:--|:--|
| Proposed model, video accuracy | 91.2 ± 2.1% | 90.2 ± 1.0%. Folds 1 and 2 match exactly; fold 3 differs by one of 33 videos. |
| Non-temporal baselines | 75.0 / 75.0 / 84.7% | identical |
| Safety gate, 6/8 at p ≥ 0.7 | 82.1% executed, 0.030 wrong actions, 3.0 s | identical |
| Proposed model at 70° elevation | 86.0% | 90.2% |

Both sets of numbers are in [`results/paper.json`](results/paper.json) and [`results/reproduction.json`](results/reproduction.json).

</details>

## Limitations

- **Offline only.** All evaluation is on recorded video. There was no flight test, and closed-loop latency was not measured.
- **Simulated viewpoint.** The dataset was filmed at low altitude, so elevation, roll and keypoint degradation were simulated on keypoints.
- **No "other movement" class, and one person only.** Everyday motion is guarded against only by the attention signal, and there is no way to pick out an authorised signaller among several people.
- **Hand shape is not used.** Signals that differ mainly in the hands, such as *not clear* and *all clear*, are confused.

<details>
<summary><b>Full list</b></summary>

<br>

- Lighting, weather, smoke, rotor downwash, long range and thermal imagery affect the pose estimator and were not evaluated.
- The elevation model is first-order and ignores depth.
- The dataset has 11 actor IDs from 8 people at one site, and each result comes from one training run per fold.
- The signals are meant for trained responders. People in distress use different ground-to-air conventions.
- No study was carried out with responders on whether they would adopt this signal vocabulary.

</details>

## Citation

If you use this code, please cite the paper. The IEEE Xplore link will be added once the proceedings are published.

```bibtex
@inproceedings{muhumed2026viewpoint,
  title     = {Toward Viewpoint-Robust Body-Signal Recognition for Commanding {UAVs} in Search and Rescue},
  author    = {Muhumed, Abdirisak Mubarik and Sariff, Nohaidda and Saruchi, Sarah 'Atifah and Mohd Tahir, Noor Idayu},
  booktitle = {IEEE International Conference on Underwater System Technology: Theory and Applications (USYS)},
  year      = {2026}
}
```

<details>
<summary><b>Dataset citation</b></summary>

```bibtex
@inproceedings{perera2018uavgesture,
  title     = {{UAV-GESTURE}: A Dataset for {UAV} Control and Gesture Recognition},
  author    = {Perera, Asanka G. and Law, Yee Wei and Chahl, Javaan},
  booktitle = {European Conference on Computer Vision Workshops (ECCVW)},
  pages     = {117--128},
  year      = {2018}
}
```

</details>

## Acknowledgements

This work builds on the first author's undergraduate final-year project at UCSI University, Malaysia. We thank A. G. Perera, Y. W. Law and J. Chahl for releasing UAV-GESTURE and its keypoints. The demo's reference poses are derived from those keypoints. The code is released under the [MIT License](LICENSE); the UAV-GESTURE data has its own terms, set by its authors.
