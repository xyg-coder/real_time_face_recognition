import click
import apis
import cv2
import numpy as np
import os
import glob

@click.group()
def recognize():
    pass

@click.command()
@click.argument('video-location')
@click.option('--saved-feature-loc', default='data/trained_features', help='the folder to save trained feature')
@click.option('--save-result-loc', default='data/result_videos', help='the folder we save result video')
@click.option('--most-frames', default=None, help='maximum frames to process and save, None means process all')
def recognize_faces_in_video(video_location, saved_feature_loc, save_result_loc, most_frames):
    """
    Firstly read all features from feature folder. Detect all faces of one video.
    Compare the face with faces in the feature folder. If recognized, label the name on the video.
    Or label unknown. Output the labeled video to result folder
    """
    def get_username(full_path):
        """
        get username from full-path
        """
        basename = os.path.basename(full_path)
        return basename.split('.')[0]
    feature_files = glob.glob(os.path.join(saved_feature_loc, '*.npy'))
    features = [np.load(feature_file) for feature_file in feature_files]
    usernames = [get_username(feature_file) for feature_file in feature_files]

    # begin processing video
    cap = cv2.VideoCapture(video_location)
    if not cap.isOpened():
        click.echo('cannot open this video', err=True)
        return
    # get the frame count of this video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # create video writer
    output_movie = cv2.VideoWriter(os.path.join(save_result_loc, os.path.basename(video_location)), fourcc, fps, (int(width), int(height)))
    frame_number = 0
    if most_frames is None:
        most_frames = video_length
    else:
        most_frames = min(int(most_frames), video_length)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_number = frame_number + 1
        rgb_frame = frame[:, :, ::-1]
        face_info_tuple = apis.recognize_faces_in_images(rgb_frame, features)
        # draw rectangles
        for top, right, bottom, left, username in face_info_tuple:
            if username is None:
                username = 'unknown'
            else:
                username = usernames[username]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, username, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        output_movie.write(frame)
        click.echo('finish processing frame {}'.format(frame_number))
        if (frame_number > most_frames - 1):
            cap.release()
            click.echo('finish processing')
            break
    output_movie.release()


recognize.add_command(recognize_faces_in_video)

if __name__ == "__main__":
    recognize()