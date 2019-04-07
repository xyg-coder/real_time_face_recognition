import click
import apis
import cv2
import time
import numpy as np

@click.group()
def train():
    pass

@click.command()
@click.argument('video_location')
@click.argument('username')
@click.option('--frame-count', default=100, help='number of frames to calculate')
def train_with_video(video_location, username, frame_count):
    cap = cv2.VideoCapture(video_location)
    if not cap.isOpened():
        click.echo('cannot open this video', err=True)
        return
    # get the frame count of this video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_count = min(frame_count, video_length)
    # how many frames we pick one frame
    seperate = int(video_length / frame_count)
    frame_list = [None] * frame_count
    result_index = 0
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_index % seperate == 0 and result_index < frame_count:
            frame_list[result_index] = frame
            result_index = result_index + 1
        frame_index = frame_index + 1
        # if there is no more frames left
        if frame_index >= video_length:
            cap.release()
            break
    # if the frame_list is not filled
    if not result_index == frame_count:
        frame_list = frame_list[:result_index]
    click.echo('begin training in process pool')
    train_with_frame_list(frame_list)
    

def face_encoding(frame):
    """
    get face encoding number with specific frame

    :param frame: frame we train
    :return: numpy array
    """
    encoding_res = apis.face_encodings(frame)
    # we only consider one face
    if not len(encoding_res) == 1:
        return None
    return encoding_res[0]


def train_with_frame_list(frame_list):
    """
    train the frame_list

    :param frame_list: list of frames(image) to train
    :return: one encoding feature
    """
    start = time.time()
    features = [face_encoding(frame) for frame in frame_list]
    features = [feature for feature in features if feature is not None]
    click.echo('process time: {}'.format(time.time() - start))
    click.echo('feature count: {}'.format(len(features)))
    click.echo('begin ransac')
    apis.ransac_mean(np.array(features))


train.add_command(train_with_video)

if __name__ == "__main__":
    train()