# Body-Signal Recognition for Commanding Drones in Search and Rescue

In search and rescue, a responder on the ground may need to direct a drone without a radio or controller. This system lets them use the body signals ground crews already use to guide aircraft, recognised through the drone's own camera. It keeps working when the drone looks down at steep angles, and acts only when it is sure.

<img src="assets/demo.png" alt="The live demo: guidance on the left, the tracked skeleton in the centre, the recognised signal and the safety gate's state on the right." width="100%">

<sub>The live demo: the signal to perform, the tracked body, and what the model recognises.</sub>

[![IEEE USYS 2026](https://img.shields.io/badge/IEEE-USYS%202026-1c1c1e?style=flat-square)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-1c1c1e?style=flat-square)](LICENSE)

## Overview

During a rescue, a responder may need a drone to hold position over a casualty, change altitude, land or abort, often with their hands busy, gloves on, or no controller within reach. Aviation already has a vocabulary for this: the standardised aircraft-handling signals that ground crews use to direct aircraft. This project teaches a drone to read those signals from its own camera.

To be useful in the field, the system has to recognise people it has never seen, keep working as the drone's viewpoint changes, never act on a misread signal, and run on the drone's own hardware. Each of these is handled by one part of a deliberately small design. Describing the body by its own keypoints lets a compact recurrent network generalise to new people. Training on simulated drone viewpoints keeps it accurate when the camera looks steeply down. A safety gate turns signals into commands only when the evidence is consistent. And because the whole chain works on 14 keypoints rather than images, it needs only milliseconds of a single CPU core.

<img src="assets/highlights.png" alt="91.2% video accuracy on unseen people; 86.0% at 70° camera elevation versus 38.4% unaugmented; 160 times fewer wrong actions with the safety gate; 2.8 ms per decision on one CPU thread." width="100%">

<sub>One number for each requirement, in the same order. Offline evaluation on recorded aerial video, tested on people the model never saw in training.</sub>

## How it works

The system is a chain of small, independent modules, joined by one simple interface: 14 body keypoints per frame. A pose estimator turns each camera frame into keypoints. Everything after that point works on the skeleton alone, which separates seeing the person from understanding the signal. The pose estimator can be swapped, the recogniser stays light enough for on-board hardware, and each stage can be tested on its own. A second idea runs through training: the recogniser never sees drone footage from steep angles, so it is taught what such footage would look like.

### The pipeline

<img src="assets/pipeline.png" alt="Pipeline: camera stream, pose estimation, body-centred features, temporal model, safety gate." width="100%">

<sub>From camera frame to command, in five stages.</sub>

1. **Camera stream.** The drone's camera delivers video at 25 frames per second.

2. **Pose estimation.** A 2D pose estimator finds 14 body joints in each frame. Any estimator works: the paper uses the OpenPose keypoints released with the dataset, and the demo uses MediaPipe.

3. **Body-centred features.** Each 1.2-second window is centred on the neck and scaled by torso length, so where the person stands in the frame, and how far away, no longer matters. Missing joints are filled in over time and joint velocities are added.

4. **Temporal model.** A two-layer LSTM reads the window and classifies it as one of 13 signals: *all clear, have command, hover, land, landing direction, move ahead, move downward, move to left, move to right, move upward, not clear, slow down* and *wave off*. It reads time in one direction only, so on board it can update frame by frame.

5. **Safety gate.** Every 0.2 seconds, the gate accepts a signal only when it wins 6 of the last 8 windows with mean confidence of at least 0.7. The responder first signals *have command* to arm the gate, and the next accepted signal is executed. *Wave off* always goes through, so an abort is never delayed.

### Viewpoint augmentation

The training videos were filmed by a drone hovering at low altitude in front of the person. In a real search, the drone is higher and looks down, its frame tilts as it moves, and the pose estimator misses joints or places them imprecisely. Rather than collect new footage for every condition, each training window is transformed into how a drone might have seen it. The camera is tilted down by up to 60° and rolled by up to 10°, joints are removed and jittered, one arm is hidden for up to 30 frames, and the pose rate is lowered to as little as 5 fps. A fresh variation is drawn every epoch, and nothing is changed at test time.

<img src="assets/augmentation.png" alt="Viewpoint augmentation of one recorded pose: elevation 60°, roll 10°, dropout and noise." width="100%">

<sub>One recorded pose and three of its training variations. The grey skeleton is the original.</sub>

## Results

The system was evaluated on [UAV-GESTURE](https://asankagp.github.io/uavgesture), a public dataset of 119 videos recorded from a hovering drone, in which eight people, filmed under 11 actor IDs, perform the 13 signals. Every result is measured on people the model never saw in training: the actor IDs are split into three groups, and each group is held out once while the model is trained on the rest ([Evaluation protocol](#evaluation-protocol)). The evaluation asks three questions in turn. How accurately are signals recognised from keypoints alone? How well does recognition hold up under the conditions a drone would face in the field? And how often would the drone act wrongly once predictions become commands? All three are answered offline, on recorded video, and every number can be reproduced with the scripts in this repository.

### Recognition

Describing keypoints relative to the body is what makes the task learnable from a moving camera: the same network goes from 15.8% to 82.9%. Velocities and augmentation then bring a keypoint-only model to 91.2%, level with a baseline that also uses colour images and optical flow.

<img src="assets/ablation.png" alt="Video accuracy for each model, from 15.8% for image coordinates to 91.2% for the proposed model." width="100%">

<sub>Video accuracy, mean of three folds. The dashed line is the RGB and optical-flow baseline reported with the dataset.</sub>

### Robustness

Viewpoint is the main way recognition fails. As the simulated camera tilts down to 70°, models trained without augmentation fall to between 21% and 38%, while the proposed model holds at 86%. Missing joints and low pose rates barely matter. Heavy keypoint noise is the remaining weakness.

<img src="assets/robustness.png" alt="Accuracy against camera elevation, roll, missing joints, keypoint noise and pose rate." width="100%">

<sub>Video accuracy as each condition is applied to test keypoints, without retraining. Training covered elevation up to 60°, roll up to 10°, dropout up to 30% and noise up to 0.08 torso lengths.</sub>

### Safety

Acting on every prediction executes the intended command almost every time, but adds nearly five wrong actions per stream. The gate cuts wrong actions to 0.03, about 160 times fewer, at the cost of 1 s of extra latency and some missed commands.

<img src="assets/safety_gate.png" alt="Executed commands and wrong actions per stream for four gate policies." width="100%">

<sub>168 command streams from unseen actors. k/n is the number of windows that must agree out of the last n, and p is the confidence threshold. The settings were chosen with the test streams in view, so read them as a trade-off rather than a tuned optimum.</sub>

## Limitations

These results describe a method evaluated offline, not a system proven in flight.

- **Offline evaluation.** All experiments use recorded video. There was no flight test, and closed-loop latency was not measured.
- **Simulated viewpoint.** The dataset was filmed at low altitude, so elevation, roll and keypoint degradation were simulated on keypoints with a first-order model that ignores depth.
- **Conditions not tested.** Lighting, weather, smoke, rotor downwash, long range and thermal imagery all act on the pose estimator, and none were evaluated.
- **No "other movement" class.** The model always outputs one of the 13 signals. The attention signal is the only guard against everyday motion being read as a command, and there is no way yet to pick out an authorised signaller among several people.
- **Hand shape is not used.** Signals that differ mainly in the hands are confused. *Not clear* is mistaken for *all clear* in 4 of 9 videos.
- **Small dataset.** UAV-GESTURE has 11 actor IDs from 8 people at one site, and each result comes from one training run per fold.
- **Intended users.** The signals are meant for trained responders. People in distress use different ground-to-air conventions.

---

## Getting started

You need Python 3.9 or later. No GPU is required: every experiment was run on an ordinary CPU.

```bash
git clone https://github.com/iamabdirazak/uav-body-signals.git
cd uav-body-signals
pip install -r requirements.txt
python -m pytest tests
```

The tests need no data and finish in seconds. To run the experiments, also download the UAV-GESTURE keypoints and splits (16 MB; the videos are not needed):

```bash
bash scripts/get_data.sh
```

<details>
<summary><b>Repository layout</b></summary>

<br>

| Path | Contents |
|:--|:--|
| `uavsignals/` | The library: [`features`](uavsignals/features.py), [`augment`](uavsignals/augment.py), [`models`](uavsignals/models.py), [`gate`](uavsignals/gate.py), [`recogniser`](uavsignals/recogniser.py), plus data loading, folds and training |
| `experiments/` | One script per result in the paper, and the figure generator |
| `demo/` | Live webcam demo: camera, pose estimation and the window |
| `weights/` | The released model, trained on all actors |
| `results/` | Every number in the paper, and a re-run on another machine |
| `scripts/` | Data download |
| `tests/` | Fast checks that need no dataset |

</details>

## Live demo

The demo runs the full pipeline on a webcam or a video file.

```bash
pip install -r demo/requirements.txt
python demo/run.py                    # or --camera 1, or --video clip.mp4
```

Stand 2 to 4 m from the camera with your whole body in view. Perform *have command* until Output shows **Ready**, then perform the signal shown in Guide.

| Key | Action |
|:--|:--|
| N / P | Next or previous signal |
| V | Switch between pose and camera view |
| R | Reset |
| Q | Quit |

The model was trained on OpenPose keypoints seen from a drone, while the demo uses MediaPipe on a ground-level camera. Treat it as a demonstration rather than an evaluation.

## Use it in your own system

`Recogniser` is the single entry point. Feed it one frame of keypoints at a time, from any pose estimator at any frame rate. It resamples to 25 Hz, classifies every 0.2 s, applies the safety gate and returns a command when one is accepted.

```python
from uavsignals import Recogniser

recogniser = Recogniser.from_pretrained()

for t, keypoints, visible in pose_stream():        # seconds, (14, 2) pixels, (14,) bool
    out = recogniser.push(t, keypoints, visible)
    if out and out.command:
        send_to_flight_controller(out.command)     # e.g. 'hover', 'land', 'wave_off'
```

Each result also carries the class probabilities (`out.probs`), the most likely signal (`out.signal`) and the gate's state (`out.armed`).

- **Keypoints.** Supply 14 joints in COCO body order without eyes and ears: nose and neck; right shoulder, elbow and wrist; left shoulder, elbow and wrist; right hip, knee and ankle; left hip, knee and ankle. If your estimator has no neck joint, use the midpoint of the shoulders.
- **Frame rate.** The pose estimator sets the frame rate, not the classifier, which needs about 3 ms per decision. Accuracy holds down to 3 fps. Run `python -m experiments.benchmark` to measure your device.
- **Gate settings.** Pass `k`, `n` and `tau` to `from_pretrained` to make the gate stricter or more responsive.
- **Retraining.** For a different pose estimator or new signals, retrain with [`training.py`](uavsignals/training.py). The released model was trained on all 119 videos, so its own accuracy can't be measured on this dataset; the fold results describe the method.

## Reproduce the paper

Each result in the paper has one command. Scripts write their numbers to `runs/results/` and their checkpoints to `runs/checkpoints/`.

| Result | Command | Time |
|:--|:--|:--|
| Recognition, all models | `python -m experiments.train` | ~45 min |
| Recognition, proposed model only | `python -m experiments.train lstm` | ~7 min |
| Non-temporal baselines | `python -m experiments.baselines` | ~4 min |
| Robustness | `python -m experiments.robustness` | ~1 min |
| Safety gate | `python -m experiments.safety_gate` | seconds |
| Model size and speed | `python -m experiments.benchmark` | seconds |
| Figures from your runs | `python -m experiments.figures --from-runs` | seconds |

<sub>Times are for a 2-core CPU. Robustness and the safety gate use the checkpoints written by training.</sub>

Training is seeded, but results still shift slightly with the PyTorch version and the processor. As a check, the code was re-run on a different machine from the paper's (Linux, 2 CPU cores, PyTorch 2.15):

| Result | Paper | Re-run |
|:--|:--|:--|
| Proposed model, video accuracy | 91.2 ± 2.1% | 90.2 ± 1.0% |
| Non-temporal baselines | 75.0, 75.0, 84.7% | Identical |
| Safety gate, 6/8 at p ≥ 0.7 | 82.1% executed, 0.030 wrong, 3.0 s | Identical |
| Proposed model at 70° elevation | 86.0% | 90.2% |

<sub>For the proposed model, folds 1 and 2 match exactly and fold 3 differs by one of 33 test videos. Both sets of numbers are in [`results/`](results).</sub>

### Evaluation protocol

All results use three actor-disjoint folds, with test actors {S13, S14, S15}, {S1, S3, S4} and {S5, S6, S9}. Folds 2 and 3 are the dataset's official splits. Official split 1 lists actor S12 in both training and test, so all S12 videos are moved to training ([`protocol.py`](uavsignals/protocol.py)). Video accuracy averages the window probabilities over each clip. Each configuration is trained once per fold with seed 0, and no hyperparameters were tuned.

## Citation

This repository implements the paper *Toward Viewpoint-Robust Body-Signal Recognition for Commanding UAVs in Search and Rescue*, presented at IEEE USYS 2026. If you use this code, please cite it. The IEEE Xplore link will be added once the proceedings are published.

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

This work builds on the first author's undergraduate final-year project at UCSI University, Malaysia. We thank A. G. Perera, Y. W. Law and J. Chahl for releasing UAV-GESTURE and its keypoints, from which the demo's reference poses are also derived.

The code is released under the [MIT License](LICENSE). The UAV-GESTURE data has its own terms, set by its authors.
