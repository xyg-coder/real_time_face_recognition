# real_time_face_recognition

This project can be divided into two steps:
1. Get static training model.
2. Use training model to recognize names in videos.

## Dockerfile

* The `Dockerfile.gpu` contains one Dockerfile which can run the `dlib` library to detect and recognize faces.

## Training Step

### Main Idea

* Use Ransac to train the face model.

### How to train

```bash
python3 train_cli.py train-with-video <your-video-location> <username>
```

## Detecting Step

### Main Idea

* Detect and encode faces in the frames and save result videos.

### How to recognize

```python
python3 recognize_cli.py recognize-faces-in-video <your-video-location>
```

## Doc

* There is one detail report in `doc` folder.

## Bash script example

* `test_experiment_cli.sh`: Detect faces in the video and save as images
* `test_recognize_cli.sh`: Recognize faces in the video and output result video
* `test_recognize_faces_in_images.sh`: Recognize faces in the image
* `test_train_cli.sh`: Train with video