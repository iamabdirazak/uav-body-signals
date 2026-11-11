# Body-signal recognition for UAVs

Direct a drone with standard aircraft-handling signals, recognised from body keypoints and robust to the drone's viewpoint.

[![IEEE USYS 2026](https://img.shields.io/badge/IEEE-USYS%202026-1c1c1e?style=flat-square)](#citation)
[![Python](https://img.shields.io/badge/Python-3.9%2B-1c1c1e?style=flat-square)](#install)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-1c1c1e?style=flat-square)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-1c1c1e?style=flat-square)](LICENSE)

<img src="assets/demo.png" alt="The live demo: guidance on the left, the tracked skeleton in the centre, the recognised signal and the safety gate's state on the right." width="100%">

<br>

This repository is the implementation behind the paper *Toward Viewpoint-Robust Body-Signal Recognition for Commanding UAVs in Search and Rescue* (IEEE USYS 2026). It contains the recognition library, every experiment in the paper, the trained model and a live webcam demo. The paper gives the full motivation, related work and discussion; this page covers what you need to use, check and extend the code.

## Overview

In search and rescue, a responder on the ground may need to tell a drone to hold position, change altitude, land or abort when a radio or handheld controller isn't practical. Aviation already has a body-signal vocabulary for this: the aircraft-handling signals that ground crews use to direct aircraft. Recognising them from a drone's own camera is hard for three reasons. The camera moves, so the person drifts around the frame. The drone looks down at changing angles. And the pose estimator misses joints.

The method here works on 2D body keypoints only, so it runs alongside any pose estimator:

- It expresses each 1.2 s window of keypoints relative to the body, so where the person is in the frame no longer matters.
- A small two-layer LSTM classifies the window into one of 13 signals.
- Training uses viewpoint augmentation that simulates camera elevation, roll, missing joints, noise and low pose rates.
- A streaming safety gate decides when a recognised signal becomes a command.

<img src="assets/highlights.png" alt="91.2% video accuracy on unseen people; 86.0% at 70° camera elevation versus 38.4% unaugmented; 160 times fewer wrong actions with the safety gate; 2.8 ms per decision on one CPU thread." width="100%">

All results come from offline evaluation on recorded aerial video. See [Limitations](#limitations).

## How it works

<img src="assets/pipeline.png" alt="Pipeline: camera stream, pose estimation, body-centred features, temporal model, safety gate." width="100%">

| Stage | What happens | Code |
|:--|:--|:--|
| Pose estimation | Any 2D estimator gives 14 body joints per frame. The paper uses the OpenPose keypoints released with the dataset; the demo uses MediaPipe. | [`demo/pose.py`](demo/pose.py) |
| Body-centred features | Missing joints are interpolated over time. Positions are centred on the neck and divided by torso length, and velocities are added, giving 56 features per frame. | [`features.py`](uavsignals/features.py) |
| Temporal model | LayerNorm and a 2 × 128 unidirectional LSTM classify 30 frames (1.2 s at 25 Hz). Because the LSTM is unidirectional, it can update frame by frame on board. | [`models.py`](uavsignals/models.py) |
| Viewpoint augmentation | During training only, windows are perturbed with camera elevation of 0–60°, roll of ±10°, anisotropic scaling, keypoint noise, joint dropout, arm occlusion and pose rates down to 5 fps. | [`augment.py`](uavsignals/augment.py) |
| Safety gate | Every 0.2 s, a signal is accepted only if it wins 6 of the last 8 windows with mean confidence of at least 0.7. *Have command* arms the gate, the next accepted signal is executed, and *wave off* always goes through. | [`gate.py`](uavsignals/gate.py) |

<img src="assets/augmentation.png" alt="Viewpoint augmentation of one recorded pose: elevation 60°, roll 10°, dropout with noise." width="72%">

The 13 signals come from the NATOPS aircraft-handling manual, as recorded in [UAV-GESTURE](https://asankagp.github.io/uavgesture): *all clear, have command, hover, land, landing direction, move ahead, move downward, move to left, move to right, move upward, not clear, slow down* and *wave off*.

## Repository

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

## Getting started

### Install

You need Python 3.9 or later. No GPU is required: every experiment in the paper was run on a laptop CPU.

```bash
git clone https://github.com/<your-username>/uav-body-signals.git
cd uav-body-signals
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data

The experiments use the OpenPose keypoints and splits that the UAV-GESTURE authors publish with their code, about 16 MB. The videos are not needed.

```bash
bash scripts/get_data.sh            # into data/UAV-GESTURE
```

To keep the data elsewhere, set `UAVG_ROOT` to its `UAVGESTURE` folder, or pass `--data` to any experiment.

### Check the install

```bash
python -m pytest tests
```

## Use the model

`Recogniser` takes one frame of keypoints at a time, from any pose estimator at any frame rate. It resamples the stream to 25 Hz internally, classifies every 0.2 s and applies the safety gate.

```python
from uavsignals import Recogniser

recogniser = Recogniser.from_pretrained()          # weights/uavsignals_lstm.pt

for t, keypoints, visible in pose_stream():        # seconds, (14, 2) pixels, (14,) bool
    out = recogniser.push(t, keypoints, visible)
    if out is None:
        continue                                   # still buffering, or the torso is not visible
    print(out.signal, out.probs.max(), 'armed' if out.armed else 'standby')
    if out.command:
        send_to_flight_controller(out.command)     # e.g. 'hover', 'land', 'wave_off'
```

Keypoints follow the COCO body order without eyes and ears: nose, neck, right shoulder, elbow and wrist, left shoulder, elbow and wrist, right hip, knee and ankle, then left hip, knee and ankle. If your estimator has no neck joint, use the midpoint of the shoulders, as [`demo/pose.py`](demo/pose.py) does for MediaPipe. To change how strict the gate is, pass its settings: `Recogniser.from_pretrained(k=3, n=4, tau=0.8)`.

The released model was trained on all 119 videos. It suits demos and integration work, but its accuracy can't be measured on UAV-GESTURE because no actor is held out. To retrain it, run `python -m experiments.release_model`.

## Live demo

The demo runs the full pipeline on a webcam or a video file, using MediaPipe for pose estimation.

```bash
pip install -r demo/requirements.txt
python demo/run.py                    # or --camera 1, or --video clip.mp4
```

Stand 2 to 4 m from the camera with your whole body in view. Perform *have command* until Output shows **Ready**, then perform the signal shown in Guide. The keys are N and P for the next and previous signal, V to switch between pose and camera view, R to reset, and Q to quit. The window follows your system's light or dark appearance.

The model was trained on OpenPose keypoints seen from a drone, while the demo uses MediaPipe on a ground-level camera, so treat it as a demonstration rather than an evaluation.

## Reproduce the paper

Each result has one command. Scripts write JSON to `runs/results/` and checkpoints to `runs/checkpoints/`. Run them from the repository root.

| Result | Command | Time* |
|:--|:--|:--|
| Recognition on unseen actors (Table I, Fig. 3) | `python -m experiments.train` | ~45 min |
| Proposed model only | `python -m experiments.train lstm` | ~7 min |
| Non-temporal baselines | `python -m experiments.baselines` | ~4 min |
| Robustness to deployment conditions (Fig. 4) | `python -m experiments.robustness` | ~1 min |
| Safety gate on command streams (Fig. 5) | `python -m experiments.safety_gate` | seconds |
| Model size and inference time | `python -m experiments.benchmark` | seconds |
| Figures on this page | `python -m experiments.figures --from-runs` | seconds |

<sub>*On a 2-core cloud CPU, no GPU. `robustness` and `safety_gate` use the checkpoints written by `train`.</sub>

**Protocol.** All results use three actor-disjoint folds, with test actors {S13, S14, S15}, {S1, S3, S4} and {S5, S6, S9}. Folds 2 and 3 are the dataset's official splits. Official split 1 lists actor S12 in both its training and test lists, so all S12 videos are moved to training ([`protocol.py`](uavsignals/protocol.py)). Video accuracy averages window probabilities over each clip. Each configuration is trained once per fold with seed 0, and no hyperparameters were tuned.

**Expected variation.** Training is seeded, but results still depend slightly on the PyTorch version and the processor. As a check, this code was re-run on a different machine from the paper's (Linux, 2 CPU cores, PyTorch 2.15):

| | Paper | Re-run |
|:--|:--|:--|
| Proposed model, video accuracy | 91.2 ± 2.1% | 90.2 ± 1.0%. Folds 1 and 2 match exactly; fold 3 differs by one of 33 videos. |
| Non-temporal baselines | 75.0 / 75.0 / 84.7% | identical |
| Safety gate, 6/8 at p ≥ 0.7 | 82.1% executed, 0.030 wrong actions, 3.0 s | identical |
| Proposed model at 70° camera elevation | 86.0% | 90.2% |

Both sets of numbers are in [`results/`](results): [`paper.json`](results/paper.json) and [`reproduction.json`](results/reproduction.json).

## Results

**Recognition on unseen actors.** Expressing keypoints relative to the body is what makes the task learnable from a moving camera. Augmentation then brings a keypoint-only model level with an RGB and optical-flow baseline.

<img src="assets/ablation.png" alt="Video accuracy for each model, from 15.8% for image coordinates to 91.2% for the proposed model." width="72%">

**Robustness.** These conditions are applied to test keypoints without retraining. Viewpoint is the main failure mode, and augmentation removes most of it. Heavy keypoint noise remains a weakness.

<img src="assets/robustness.png" alt="Accuracy against camera elevation, roll, missing joints, keypoint noise and pose rate." width="72%">

**Safety gate.** The gate trades a lower execution rate and about 1 s of extra latency for far fewer wrong actions. These operating points were computed on the test streams, so read them as a trade-off curve rather than a tuned setting.

<img src="assets/safety_gate.png" alt="Executed commands and wrong actions per stream for four gate policies." width="72%">

## Running on your hardware

- **The pose estimator sets the frame rate, not the classifier.** The classifier needs about 3 ms per decision on one CPU thread. Run `python -m experiments.benchmark` on your target device.
- **Low pose rates are fine.** Accuracy stays between 90.0% and 91.2% as the pose rate drops from 25 to 3 fps, and `Recogniser` resamples whatever rate it gets.
- **Changing the pose estimator may need retraining.** The paper's model was trained on OpenPose keypoints, and estimators such as BlazePose define joints differently. For a new platform, map its joints to the 14 above, collect a few clips per signal, and retrain with [`training.py`](uavsignals/training.py).

## Limitations

- All experiments are offline on recorded video. There was no flight test, and closed-loop latency was not measured.
- The dataset was recorded at low altitude with the drone in front of the person. Elevation, roll, dropout, noise and pose rate were simulated on keypoints, and the elevation model ignores depth.
- Lighting, weather, smoke, rotor downwash, long range and thermal imagery affect the pose estimator and were not evaluated.
- There is no class for unrelated movement. The attention signal is the only safeguard against everyday motion being read as a command.
- There is no handling of several people in view, or of which person is authorised to give commands.
- Hand shape is not represented, so *not clear* and *all clear* are confused (5 of 9 *not clear* videos are correct).
- The dataset has 11 actor IDs from 8 people at one site. Results come from one training run per fold.
- The signals are meant for trained responders. People in distress use different ground-to-air conventions.

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

Please also cite UAV-GESTURE if you use the data:

```bibtex
@inproceedings{perera2018uavgesture,
  title     = {{UAV-GESTURE}: A Dataset for {UAV} Control and Gesture Recognition},
  author    = {Perera, Asanka G. and Law, Yee Wei and Chahl, Javaan},
  booktitle = {European Conference on Computer Vision Workshops (ECCVW)},
  pages     = {117--128},
  year      = {2018}
}
```

## Acknowledgements

This work builds on the first author's undergraduate final-year project at UCSI University, Malaysia. We thank A. G. Perera, Y. W. Law and J. Chahl for releasing UAV-GESTURE and its keypoints. The reference poses in the demo are derived from those keypoints, and the dataset is for academic research only.

## License

The code is released under the [MIT License](LICENSE). The UAV-GESTURE data has its own terms, set by its authors.
