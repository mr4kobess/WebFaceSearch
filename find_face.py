import numpy as np
from face_scan import FaceScan
from index import Index
import config


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



facescan = FaceScan(detector_name=config.detector_name, model_name=config.model_name)
index = Index(new=False, klusters=config.klusters, n_probe=config.n_probe, path_to_index_dir=config.path_to_index_dir)

findface = FindFace(index=index, face_scan=facescan)