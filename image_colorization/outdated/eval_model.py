from keras.models import load_model
from image_colorization.outdated.data_server import load_dataset
import cv2
import numpy as np
dataset_path = "datasets/broadleaf/"


def main():
    x_train, y_train = load_dataset(dataset_path)

    a = y_train[0, :, :, 0]
    a = 256 * a + 256 / 2
    a = a.astype(np.uint8)

    b = y_train[0, :, :, 1]
    b = 256 * b + 256 / 2
    b = b.astype(np.uint8)

    L = x_train[0, :, :, 0]
    L = 256 * L + 256 / 2
    L = L.astype(np.uint8)

    new_Lab = np.stack((L, a, b), axis=2)

    new_bgr = cv2.cvtColor(new_Lab, cv2.COLOR_LAB2BGR)
    cv2.imshow("new original", new_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    model = load_model("weights/modelV3.hdf5")
    model.load_weights('weights/weightsV3.500.hdf5')

    score = model.evaluate(x_train, y_train, verbose=1)
    ab_new = model.predict(x_train)

    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    print('\n', 'Test accuracy:', score[0])


if __name__ == "__main__":
    main()
