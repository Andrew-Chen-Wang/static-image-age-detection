"""
Module for training the known people.
"""
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def create_recognizer_with_label(embedding_path, recognizer_path, name_path) -> None:
    """
    Creates the name and recognizer pickle files from the custom embedder

    :param embedding_path: the custom embedder that was used during the
    training from the dataset, not the given model
    :param recognizer_path: the designated path for the recognizer
    :param name_path: the designated path for the labels/names
    :return: None
    """
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embedding_path, "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["encodings"], labels)

    # write the actual face recognition model to disk
    f = open(recognizer_path, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(name_path, "wb")
    f.write(pickle.dumps(le))
    f.close()
