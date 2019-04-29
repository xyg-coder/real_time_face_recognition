import click
import apis
import cv2
import numpy as np
import os
import multiprocessing

@click.group()
def experiment():
    pass

@click.command()
@click.argument('video-location')
@click.argument('output-location')
@click.option('--frame-count', default=100, help='number of frames to calculate')
def test_hog_cnn(video_location, output_location, frame_count):
    """
    This function will read one video. Split it into different frames. 
    Detect faces in these frames and output result to output folder.
    """
    cap = cv2.VideoCapture(video_location)
    if not cap.isOpened():
        click.echo('cannot open this video', err=True)
        return
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_count = min(frame_count, video_length)
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
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame_list = [frame[:, :, ::-1] for frame in frame_list]

    # firstly use cnn
    print('start using cnn to detect faces')
    cnn_faces = [detect_faces_cnn(frame, index) for index, frame in enumerate(frame_list)]
    print('finish detecting faces in cnn')

    # use hog to detect
    print('start using hog to detect faces')
    arguments = [[frame, index] for index, frame in enumerate(frame_list)]
    with multiprocessing.Pool(processes=4) as pool:
        hog_faces = pool.starmap(detect_faces_hog, arguments)

    # draw cnn rectangles
    print('begin writing to files')
    index = 0
    for faces in cnn_faces:
        frame = frame_list[index][:,:,::-1]
        for top, right, bottom, left in faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        index += 1
    index = 0
    for faces in hog_faces:
        frame = frame_list[index][:,:,::-1]
        for top, right, bottom, left in faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_location, '{}.png'.format(index)), frame)
        index += 1
    

def detect_faces_cnn(frame, index):
    print('cnn {} begins'.format(index))
    face_locations_rect = apis._raw_face_locations(frame, model='cnn')
    return [apis._rect_to_css(rect) for rect in face_locations_rect]


def detect_faces_hog(frame, index):
    print('hog {} begins'.format(index))
    face_locations_rect = apis._raw_face_locations(frame, model='hog')
    return [apis._rect_to_css(rect) for rect in face_locations_rect]

experiment.add_command(test_hog_cnn)


if __name__ == "__main__":
    experiment()