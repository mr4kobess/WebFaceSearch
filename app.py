from flask import Flask
from flask import request
from find_face import findface

app = Flask(__name__)


@app.route('/search', methods=['POST', 'GET'])
def search():
    faces_ids = dict()
    count_neighbors = int(request.form.get('count_neighbors', 3))
    many_faces = int(request.form.get('many_faces', 0))
    for k, f in request.files.items():
        img = f.read()
        find_indexes = findface.search(img, n=count_neighbors, many=many_faces)
        faces_ids[k] = find_indexes
    return faces_ids


if __name__ == '__main__':
    app.run(debug=True, port=9088, host='0.0.0.0')
