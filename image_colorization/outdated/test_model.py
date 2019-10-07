import numpy as np
import cv2
import glob
from keras.models import load_model

img_size = 64

dataset_path = "datasets/broadleaf/"


def main():

    model = load_model("model_states/modelV3.hdf5")
    model.load_weights('model_states/weightsV3.500.hdf5')

    for img_name in glob.glob(dataset_path + "*"):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (img_size, img_size))

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.stack((gray, gray, gray), axis=2)

        Lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = Lab_img[:, :, 0]
        ab = Lab_img[:, :, 1:3]

        L_input = np.reshape(L, (1, img_size, img_size, 1))

        L_input = (L_input - 256 / 2) / 256

        ab_new = model.predict(L_input)

        score = model.evaluate(L_input, np.reshape(ab, (1, img_size, img_size, ab.shape[2])), verbose=1)
        print(score)
        a = ab_new[0, :, :, 0]
        a = 256*a + 256/2

        b = ab_new[0, :, :, 1]
        b = 256*b + 256/2

        new_Lab = np.stack((L, a, b), axis=2)

        new_bgr = cv2.cvtColor(new_Lab, cv2.COLOR_LAB2BGR)
        cv2.imshow("new original", new_bgr)
        cv2.imshow("original", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
