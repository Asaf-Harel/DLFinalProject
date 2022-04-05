import sys
import os
from os import path

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QLabel, QPushButton
from PyQt5.QtGui import QFont, QPixmap, QColor

import numpy as np
from binvis.converter import convert_to_image
import utils

import string
import random


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__model = utils.create_model('weights/weights.h5')  # Load model
        self.setWindowTitle("Antivirus")
        self.setFixedSize(1200, 720)
        self.initUI()

        self.__classes = ['malware ❌', 'normal ✅']

        # Check which OS is running for later use in the file dialog
        if os.name == 'posix':
            self.__home_path = path.expanduser('~')
        else:
            self.__home_path = os.environ['USERPROFILE']

    def initUI(self):
        """Create all the necessary UI elemnts"""
        title_font = QFont("Arial", 48)
        title_font.setBold(True)

        self.title = QLabel("Antivirus", self)
        self.title.setFont(title_font)
        self.title.setGeometry(495, 40, 210, 55)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(525, 165, 400, 400)

        self.file_button = QPushButton("Select File", self)
        self.file_button.setGeometry(500, 330, 200, 30)
        self.file_button.clicked.connect(self.launchFileDialog)

        self.confrim_button = QPushButton("Analyze", self)
        self.confrim_button.setGeometry(450, 630, 300, 40)
        self.confrim_button.clicked.connect(self.confirm_clicked)

        self.confrim_button.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: #000000;
                color: white;
                font-size: 20pt;
                border-radius: 10px;
            }
            
            QPushButton:disabled {
                background-color: #444444;
                color: #eeeeee;
            }
            """
        )

    def launchFileDialog(self):
        """Open File Dialog"""
        files_filter = "Executables (*.exe)"  # Show only exe files
        file_path, file_type = QFileDialog.getOpenFileName(self, caption="Select a .exe file", directory=os.path.join(
            os.path.join(self.__home_path, "Desktop")), filter=files_filter, initialFilter='Executables (*.exe)')

        malware_file_path = ''.join(random.choices(string.ascii_letters, k=16)) + ".png"  # Create random name for the image file

        self.malware_image = convert_to_image(256, file_path, malware_file_path)  # Convert file to image

        self.file_button.setGeometry(500, 120, 200, 30)

        image_pixmap = QPixmap(malware_file_path)
        self.image_label.setPixmap(image_pixmap)
        self.image_label.resize(image_pixmap.width(), image_pixmap.height())  # Show the converted file image
        os.remove(malware_file_path)  # Remove the image file

    def confirm_clicked(self):
        #  Modify image to fit into the model
        image_array = np.array(self.malware_image.resize((220, 220)))
        modified_image = utils.normalize_image(np.array([image_array]))
        
        result = self.__model.predict(modified_image).argmax(axis=1)[0]  # Predict image
        result_str = self.__classes[result]  # Convert result to user friendly string

        self.confrim_button.setEnabled(False)
        
        # Open new window with result
        msg = QMessageBox()
        msg.setWindowTitle("Analyzer")
        msg.setText(result_str)
        msg.exec_()
        self.confrim_button.setEnabled(True)


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec_())  # Run Application

