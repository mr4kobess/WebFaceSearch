import os
import json
import math
import glob
import numpy as np
import cv2
import time
import pandas as pd
import re
import pickle
import shutil
from face_scan import FaceScan
from index import Index


class FindFace:
    def __init__(self, index: Index, face_scan: FaceScan):
        self._face_scan = face_scan
        self._index = index

    def add_facies(self, data, ignore_errors=False):
        good_data, bad_data = self._face_scan.scan(data, ignore_errors=ignore_errors)
        self._index.add_vectors(good_data['vectors'], good_data['ids'])
        return good_data, bad_data

    def search(self, img_path, n=1, many=True, threshold=330):
        enc = self._face_scan.scan_img(img_path, many=many)
        if not enc.any():
            return []
        if not many:
            enc = np.expand_dims(enc, axis=0)
        find_indexes = self._index.search(enc, n)
        result = [list() for _ in range(len(find_indexes[0]))]
        for i, distances in enumerate(find_indexes[0]):
            for j, distance in enumerate(distances):
                if distance < threshold:
                    result[i].append(int(find_indexes[1][i][j]))
        return result



facescan = FaceScan(detector_name='mtcnn')
index = Index(new=False, klusters=512, n_probe=16, path_to_index_dir='/run/media/mansur/My Data/Загрузки/index')

findface = FindFace(index=index, face_scan=facescan)