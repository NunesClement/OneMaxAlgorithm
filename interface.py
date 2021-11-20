import sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton, QVBoxLayout, QComboBox, QWidget
import Lanceur


class GlobalParameter:
    def __init__(self, mutationFlipNumber, mutationFlipProbability):
        self.mutationFlipNumber = mutationFlipNumber
        self.mutationFlipProbability = mutationFlipProbability


globalState = GlobalParameter("1-flip", 0.5)


class Second(QMainWindow):
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        Lanceur.launch_the_launcher(globalState)

        class MainWindow(QMainWindow):

            def __init__(self):
                super(MainWindow, self).__init__()
                self.title = "Image Viewer"
                self.setWindowTitle(self.title)

                label = QLabel(self)
                pixmap = QPixmap('test.png')
                label.setPixmap(pixmap)
                self.setCentralWidget(label)
                self.resize(pixmap.width(), pixmap.height())

        self.w = MainWindow()
        self.w.resize(600, 600)
        self.w.show()


class First(QMainWindow):

    def __init__(self, parent=None):
        super(First, self).__init__(parent)
        self.pushButton = QPushButton("Run programme")
        self.setGeometry(500, 500, 500, 250)

        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.title = "Menu de sélection"
        self.setWindowTitle(self.title)

        self.buttonRun = QPushButton("Run")

        self.fitnessLabel = QLabel("Validation")
        self.choixFitness = QLabel("Choix fitness par encore disponible - onemax par défaut")

        self.validationLabel = QLabel("Validation")
        self.selectionChoix = QComboBox()
        self.selectionChoix.addItem("selection de deux gênomes randoms (en vu d'un croisement random)")
        self.selectionChoix.addItem("selection de deux meilleurs gênes (en vu d'un croisement random)")
        self.selectionChoix.addItem("selection pair parmis x random TODO")

        self.mutationLabel = QLabel("Mutation")

        self.mutationChoix = QComboBox()
        self.mutationChoix.addItem("1-flip")
        self.mutationChoix.addItem("2-flip")
        self.mutationChoix.addItem("3-flip")
        self.mutationChoix.addItem("4-flip")
        self.mutationChoix.addItem("5-flip")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.fitnessLabel)
        self.layout.addWidget(self.choixFitness)
        self.layout.addWidget(self.validationLabel)

        self.layout.addWidget(self.buttonRun)
        self.layout.addWidget(self.selectionChoix)
        self.layout.addWidget(self.mutationLabel)
        self.layout.addWidget(self.mutationChoix)
        self.dialogs = "test"
        self.buttonRun.clicked.connect(self.on_pushButton_clicked)
        self.mutationChoix.currentTextChanged.connect(self.setMutationFLip)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.show()

    def on_pushButton_clicked(self):
        dialog = Second(self)
        # self.dialogs.append(dialog)
        # dialog.show()

    def setMutationFLip(self, s):
        globalState.mutationFlipNumber = str(s)



def main():
    app = QApplication(sys.argv)
    main = First()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

# class MainWindow(QMainWindow):
#
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.title = "Image Viewer"
#         self.setWindowTitle(self.title)
#
#         label = QLabel(self)
#         pixmap = QPixmap('test.png')
#         label.setPixmap(pixmap)
#         self.setCentralWidget(label)
#         self.resize(pixmap.width(), pixmap.height())
#
#
# class MyWidget(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         self.title = "Menu de sélection"
#         self.setWindowTitle(self.title)
#         # self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]
#
#         self.buttonRun = QtWidgets.QPushButton("Run")
#         self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignCenter)
#
#         self.fitnessLabel = QtWidgets.QLabel("Validation")
#         self.choixFitness = QtWidgets.QLabel("Choix fitness par encore disponible - onemax par défaut")
#
#         self.validationLabel = QtWidgets.QLabel("Validation")
#
#         self.selectionChoix = QtWidgets.QComboBox()
#         self.selectionChoix.addItem("selection de deux gênomes randoms (en vu d'un croisement random)")
#         self.selectionChoix.addItem("selection de deux meilleurs gênes (en vu d'un croisement random)")
#         self.selectionChoix.addItem("selection pair parmis x random TODO")
#
#         self.mutationLabel = QtWidgets.QLabel("Mutation")
#
#         self.mutationChoix = QtWidgets.QComboBox()
#         self.mutationChoix.addItem("Mutation avec nb de mutations + proba")
#         self.mutationChoix.addItem("Mutation avec nb de mutations + proba 1/nbPop")
#
#         self.layout = QtWidgets.QVBoxLayout(self)
#         self.layout.addWidget(self.fitnessLabel)
#         self.layout.addWidget(self.choixFitness)
#         self.layout.addWidget(self.validationLabel)
#
#         self.layout.addWidget(self.buttonRun)
#         self.layout.addWidget(self.selectionChoix)
#         self.layout.addWidget(self.mutationLabel)
#         self.layout.addWidget(self.mutationChoix)
#         self.buttonRun.clicked.connect(launch(self))

#         def launch(self):
#             Lanceur.launch_the_launcher()
#
#             w = MainWindow()
#             w.resize(600, 600)
#
#             self.show()
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#
#     widget = MyWidget()
#     widget.resize(800, 600)
#     widget.show()
#     w = MainWindow()
#     w.resize(800, 600)
#
#     w.show()
#
#     sys.exit(app.exec())
