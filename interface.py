import sys
from PySide6 import QtCore, QtWidgets


def launch():
    # self.text.setText(random.choice(self.hello))
    print("Lancer")


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignCenter)

        self.fitnessLabel = QtWidgets.QLabel("Validation")
        self.choixFitness = QtWidgets.QLabel("Choix fitness par encore disponible - onemax par défaut")

        self.validationLabel = QtWidgets.QLabel("Validation")

        self.selectionChoix = QtWidgets.QComboBox()
        self.selectionChoix.addItem("selection de deux gênomes randoms (en vu d'un croisement random)")
        self.selectionChoix.addItem("selection de deux meilleurs gênes (en vu d'un croisement random)")
        self.selectionChoix.addItem("selection pair parmis x random TODO")

        self.mutationLabel = QtWidgets.QLabel("Mutation")

        self.mutationChoix = QtWidgets.QComboBox()
        self.mutationChoix.addItem("Mutation avec nb de mutations + proba")
        self.mutationChoix.addItem("Mutation avec nb de mutations + proba 1/nbPop")

        self.layout = QtWidgets.QVBoxLayout(self)
        # self.layout.addWidget(self.text)
        self.layout.addWidget(self.fitnessLabel)
        self.layout.addWidget(self.choixFitness)
        self.layout.addWidget(self.validationLabel)

        # self.layout.addWidget(self.button)
        self.layout.addWidget(self.selectionChoix)
        self.layout.addWidget(self.mutationLabel)
        self.layout.addWidget(self.mutationChoix)

        # self.button.clicked.connect(launch)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
