import os
import glob
import numpy as np
import re
import shutil
import faiss


class Index:

    def __init__(self, klusters=16, dim=512, n_probe=4, path_to_index_dir='index', new=True):

        # Создание структуры папок
        self._root_path = path_to_index_dir
        self._blocks_path = os.path.join(self._root_path, 'blocks/')
        self._trained_block_path = os.path.join(self._root_path, 'trained_block.index')
        self._ivf_data_path = os.path.join(self._root_path, 'merged_index.ivfdata')
        self._populate_path = os.path.join(self._root_path, 'populated.index')

        # Параметры
        self._k = klusters
        self._dim = dim
        self._n_probe = n_probe

        if new:
            shutil.rmtree(path_to_index_dir, ignore_errors=True)
            os.mkdir(path_to_index_dir)
            os.mkdir(self._blocks_path)
            # Пустой индекс с параметрами
            self._index = faiss.index_factory(dim, f"IVF{self._k},Flat", faiss.METRIC_L2)
            faiss.write_index(self._index, self._trained_block_path)
        else:
            self._load()

    def _get_last_block_index(self):
        blocks = glob.glob(self._blocks_path + '*')
        if not blocks:
            return 0
        last_block = blocks[-1]
        print(last_block)
        block_id = re.search(r'block_(\d+)', last_block)
        if block_id:
            return int(block_id.group(1))
        else:
            return 0

    def train(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors).astype('float32')
        self._index.train(vectors)

        # Сохраняем пустой обученный индекс, содержащий только параметры
        faiss.write_index(self._index, os.path.join(self._root_path, "trained_block.index"))

    def add_vectors(self, vectors, vectors_ids, chunksize=100000):
        assert self._index.is_trained, 'Before you need run train_index.'

        if isinstance(vectors, list):
            vectors = np.array(vectors).astype('float32')
        if isinstance(vectors_ids, list):
            vectors_ids = np.array(vectors_ids)

        first_block = 0
        last_block = len(vectors)
        last_block_id = self._get_last_block_index() + 1
        # Поочередно создаем новые индексы на основе обученного
        # Блоками добавляем в них части датасета:
        for i, bno in enumerate(range(first_block, last_block, chunksize)):
            block_vectors = vectors[bno:bno + chunksize]
            block_vectors_ids = vectors_ids[bno:bno + chunksize]  # id векторов, если необходимо
            index = faiss.read_index(self._trained_block_path)
            index.add_with_ids(block_vectors, block_vectors_ids)
            faiss.write_index(index, os.path.join(self._blocks_path, f"block_{last_block_id + i}.index"))
        self._build()
        self._load()

    def _build(self):
        ivfs = []
        for bno in glob.glob(os.path.join(self._blocks_path, 'block_*')):
            index = faiss.read_index(bno, faiss.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            # считать index и его inv_lists независимыми
            # чтобы не потерять данные во время следующей итерации:
            index.own_invlists = False
        # создаем финальный индекс:
        index = faiss.read_index(self._trained_block_path)
        # готовим финальные invlists
        # все invlists из блоков будут объединены в файл merged_index.ivfdata
        invlists = faiss.OnDiskInvertedLists(index.nlist, index.code_size, self._ivf_data_path)
        ivf_vector = faiss.InvertedListsPtrVector()
        for ivf in ivfs:
            ivf_vector.push_back(ivf)

        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
        index.ntotal = ntotal  # заменяем листы индекса на объединенные
        index.replace_invlists(invlists)
        data_path = self._populate_path
        faiss.write_index(index, data_path)  # сохраняем всё на диск

    def _load(self):
        self._index = faiss.read_index(self._populate_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        self._index.nprobe = self._n_probe

    def search(self, vector, n):
        return self._index.search(vector, n)
