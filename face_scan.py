from deepface import DeepFace
from deepface.detectors import FaceDetector
from deepface.commons import functions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


class FaceScan:
    #            faster  --------------> more accurately
    # DETECTORS:  ssd, opencv, dlib, retinaface, mtcnn

    def __init__(self, model_name='Facenet512', detector_name='dlib'):
        self._model = DeepFace.build_model(model_name)
        self._model_name = model_name

        self._detector = FaceDetector.build_model(detector_name)
        self._detector_name = detector_name

    def detect_face(self, img_str):
        nparr = np.fromstring(img_str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            return functions.preprocess_face(img_np, detector_backend=self._detector_name)[0]
        except ValueError:
            return None

    def detect_faces(self, img_str, target_size=(224, 224)):
        nparr = np.fromstring(img_str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = []
        for img, region in FaceDetector.detect_faces(self._detector, self._detector_name, img_np):
            if img.shape[0] > 0 and img.shape[1] > 0:
                factor_0 = target_size[0] / img.shape[0]
                factor_1 = target_size[1] / img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
                img = cv2.resize(img, dsize)

                # Then pad the other side to the target size by adding black pixels
                diff_0 = target_size[0] - img.shape[0]
                diff_1 = target_size[1] - img.shape[1]
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                             'constant')

            if img.shape[0:2] != target_size:
                img = cv2.resize(img, target_size)

            img_pixels = image.img_to_array(img)  # what this line doing? must?
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]
            result.append(img_pixels)
        return result

    def scan(self, facies: dict, ignore_errors=False):

        good_data = {'ids': [], 'vectors': []}
        bad_data = {'ids': [], 'paths': []}

        for id, img_path in zip(facies['ids'], facies['paths']):
            try:
                img_encodings = self.scan_img(img_path)
            except AssertionError as e:
                if ignore_errors:
                    img_encodings = np.array([])
                else:
                    raise e
            if img_encodings.any():
                good_data['vectors'].append(img_encodings)
                good_data['ids'].append(id)
            else:
                bad_data['ids'].append(id)
                bad_data['paths'].append(img_path)

        return good_data, bad_data

    def scan_img(self, img_path, many=False) -> np.array:
        if many:
            encodings = []
            faces = self.detect_faces(img_path)
            for face in faces:
                embedding = DeepFace.represent(img_path=face[0], model_name=self._model_name, model=self._model,
                                               detector_backend='skip', enforce_detection=False,
                                               normalization=self._model_name)
                encodings.append(embedding)
            return np.array(encodings).astype('float32')
        else:
            embedding = np.array([])
            face = self.detect_face(img_path)
            if face is not None:
                embedding = DeepFace.represent(img_path=face, model_name=self._model_name, model=self._model,
                                               detector_backend='skip', enforce_detection=False,
                                               normalization=self._model_name)
            return np.array(embedding).astype('float32')