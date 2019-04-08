# real_time_face_recognition

This project can be divided into two steps:
1. Get static training model.
2. Use training model to recognize names in videos.

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

## TODO

* Increase the processing speed to make the recognition real-time.