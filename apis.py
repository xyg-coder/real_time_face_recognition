import PIL.Image
import dlib
import numpy as np
import random

import models

face_detector = dlib.get_frontal_face_detector()

predictor_68_point = dlib.shape_predictor(models.shape_predictor_68_points_model_location())
predictor_5_point = dlib.shape_predictor(models.shape_predictor_5_points_model_location())
cnn_face_detector = dlib.cnn_face_detection_model_v1(models.human_face_detector_model_location())
face_encoder = dlib.face_recognition_model_v1(models.face_recognition_resnet_model_location())

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib 'rect' object

    :param css: plain tuple in (top, right, bottom, left) order
    :return: a dlib 'rect' object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _raw_face_locations(img, number_of_times_to_upsample=1, model='cnn'):
    """
    returns an array of bounding boxes of human faces in an image

    :param img: an image as a numpy array
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate 
        deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: a list of dlib 'rect' objects of found face locations
    """
    if model == 'cnn':
        rectangles = cnn_face_detector(img, number_of_times_to_upsample)
        return [rectangle.rect for rectangle in rectangles]
    else:# hog
        return face_detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image, face_locations=None, model='large'):
    """
    given an image, return a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_location: optionally provide a list of face locations to search
    :param model: 'large' (default) uses 68 points, 'small' uses 5 points which is faster
    :return: a list of dicts of face features locations
    """
    if face_locations == None:
        face_locations = _raw_face_locations(img=face_image)
        face_locations = [face_location for face_location in face_locations]
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_model = predictor_68_point
    if model == 'small':
        pose_model = predictor_5_point

    return [pose_model(face_image, face_location) for face_location in face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)   
    """
    raw_landmakrs = _raw_face_landmarks(face_image, known_face_locations, model='small')
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark)) for raw_landmark in raw_landmakrs]


def ransac_mean(features, ratio_threshold=0.9, dist_threshold=0.4):
    """
    Use idea of ransac to get one mean feature. Everytime, randomly pick one feature, and compare the distances
    with all other features. If the number of features with less distance than dist_threshold is more than ratio_threshold
    stop iterating and return the mean value of these features

    :param features: 2-d dimension numpy array, every row is feature of one face
    :param ratio_threshold: ransac inlier ratio threshold
    :param dist_threshold: within this threshold is considered inlier
    :return: return final average feature for inliers or throw one exception
    """
    feature_count = features.shape[0]
    max_iters = feature_count * 2
    for i in range(max_iters):
        pick_index = random.randint(0, feature_count - 1)
        pick_feature = features[pick_index]
        inliers = np.linalg.norm(features - pick_feature, axis=1) < dist_threshold
        inliers_count = np.sum(inliers)
        print('{}, {}'.format(inliers_count, feature_count))
        if inliers_count >= ratio_threshold * feature_count:
            inliers = features[inliers]
            print('final iterations: {}, inliers_count: {}'.format(i, inliers_count))
            return np.mean(inliers, axis=0)
    raise ValueError('the features are not stable enough')


def recognize_faces_in_images(face_image, features, dist_threshold=0.4):
    """
    detect all faces in the face_image, compare it with features.
    compare the minimum distance with the dist_threshold, if not match, unknown user

    :param face_image: input face_image
    :param features: 2-d numpy array. Every row is one feature
    :param dist_threshold: inside this threshold means match
    :return: one list of tuples representing detected face. 
        [(top, right, bottom, left, min_index)].
        If no match, min_dist_index = None
    """
    face_locations = _raw_face_locations(img=face_image)
    face_locations = [_rect_to_css(face_location) for face_location in face_locations]

    face_features = face_encodings(face_image, face_locations)
    result = [None] * len(face_features)
    result_index = 0
    for face_feature in face_features:
        distances = np.linalg.norm(face_feature - features, axis=1)
        min_dist_index = np.argmin(distances)
        if distances.ndim == 0:
            min_dist = distances
        else:
            min_dist = distances[min_dist_index]
        print(min_dist)
        if min_dist > dist_threshold:
            min_dist_index = None
        result[result_index] = (face_locations[result_index][0], face_locations[result_index][1], 
            face_locations[result_index][2], face_locations[result_index][3], min_dist_index)
        result_index = result_index + 1
    return result