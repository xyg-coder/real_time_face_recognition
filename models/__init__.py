from pkg_resources import resource_filename

def shape_predictor_5_points_model_location():
    return resource_filename(__name__, 'shape_predictor_5_face_landmarks.dat')

def shape_predictor_68_points_model_location():
    return resource_filename(__name__, 'shape_predictor_68_face_landmarks.dat')

def face_recognition_resnet_model_location():
    return resource_filename(__name__, 'dlib_face_recognition_resnet_model_v1.dat')

def human_face_detector_model_location():
    return resource_filename(__name__, 'mmod_human_face_detector.dat')