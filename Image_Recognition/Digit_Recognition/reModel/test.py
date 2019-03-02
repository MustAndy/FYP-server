import sys

from PyQt5.QtWidgets import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='4'
import predict
import GUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    predict = predict.Test_predict()
    w = GUI.Widget(predict)
    sys.exit(app.exec_())