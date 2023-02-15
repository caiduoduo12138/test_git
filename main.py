import os
import cv2
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from IPython import display
from matplotlib import pyplot as plt


def img2sk(img_floder='/home/cai/project/activate_learning/test_data/images/', label_file="/home/cai/project/activate_learning/test_data/label.txt"):
    f = open(label_file)
    lines = f.readlines()
    labels_np = np.ones((len(lines)))
    for index, line in enumerate(lines):
        labels_np[index] = int(line.strip("\n").split(" ")[-1])

    imgs_name = os.listdir(img_floder)
    imgs_np = np.ones((len(imgs_name), 224*224*3), dtype=np.uint8)
    for ind, each in enumerate(imgs_name):
        img = cv2.imread(img_floder+each)
        img_resize = cv2.resize(img, (224, 224))
        img_flatten = img_resize.reshape(-1)
        imgs_np[ind] = img_flatten
        # img_ori = img_flatten.reshape(224, 224, 3)
        # cv2.imwrite(img_floder+"1.jpg", img_ori)
    return imgs_np, labels_np

def generate_labelfile(img_path="/home/cai/project/activate_learning/test_data/"):
    class_name = ["cat", "dog"]
    f = open(img_path+"label.txt", 'w')
    for each in os.listdir(img_path):
        if each not in class_name:
            continue
        for cls in os.listdir(img_path+each+"/"):
            text = cls+" "+str(class_name.index(each))+"\n"
            f.write(text)
    f.close()

generate_labelfile()
imgs, labels = img2sk()

n_initial = 20
# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(imgs, labels)

initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_initial, y_training=y_initial
)
learner.fit(X_initial, y_initial)

n_queries = 5

for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    ori_img = query_inst[0].reshape(224, 224, 3)
    cv2.imshow("img", ori_img)
    cv2.waitKey(1000)
    print("please enter a value:")
    y_new = np.array([int(float(input()))], dtype=int)
    learner.teach(query_inst.reshape(1, -1), y_new)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    cv2.destroyAllWindows()
pred = learner.predict(X_test)

correct = 0.0
for i in range(pred.shape[0]):
    if int(y_test[i]) == int(pred[i]):
        correct += 1
print("acc:{}".format(correct/pred.shape[0]))

