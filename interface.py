import sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton, QVBoxLayout, QComboBox, QWidget, QLineEdit
import Lanceur


class GlobalParameter:
    def __init__(self, seed, taille_pop, mutation_params, selection_params,
                 fitness_limit, generation_limit, genome_length):
        self.seed = seed
        self.mutation_params = mutation_params
        self.selection_params = selection_params
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit
        self.genome_length = genome_length
        self.taille_pop = taille_pop


globalState = GlobalParameter(13, 10,
                              ["1-flip", 0.5],
                              "selection_pair_parmis_s_random",
                              1000, 1000, 100)


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
        self.mutationChoix.addItem("bitflip")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.fitnessLabel)
        self.layout.addWidget(self.choixFitness)
        self.layout.addWidget(self.validationLabel)

        self.layout.addWidget(self.buttonRun)
        self.layout.addWidget(self.selectionChoix)
        self.layout.addWidget(self.mutationLabel)
        self.layout.addWidget(self.mutationChoix)

        self.sizePopLabel = QLabel("nb d'individus")
        self.sizePop = QLineEdit()
        self.sizePop.setText("10")
        self.layout.addWidget(self.sizePopLabel)
        self.layout.addWidget(self.sizePop)
        self.sizePop.textChanged.connect(self.change_size_pop)

        self.fitnessMaxLabel = QLabel("set fitness max")
        self.fitnessMax = QLineEdit()
        self.fitnessMax.setText("25000")
        self.layout.addWidget(self.fitnessMaxLabel)
        self.layout.addWidget(self.fitnessMax)
        self.fitnessMax.textChanged.connect(self.change_fitness_max)

        self.genomeTailleLabel = QLabel("genome taille")
        self.genomeTaille = QLineEdit()
        self.genomeTaille.setText("100")
        self.layout.addWidget(self.genomeTailleLabel)
        self.layout.addWidget(self.genomeTaille)
        self.genomeTaille.textChanged.connect(self.change_genome_taille_label)

        self.generationNbLabel = QLabel("Itération / génération")
        self.generationNb = QLineEdit()
        self.generationNb.setText("1000")
        self.layout.addWidget(self.generationNbLabel)
        self.layout.addWidget(self.generationNb)
        self.generationNb.textChanged.connect(self.change_nb_generation)

        self.dialogs = ""
        self.buttonRun.clicked.connect(self.on_pushButton_clicked)
        self.mutationChoix.currentTextChanged.connect(self.setMutationFLip)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.show()

    def change_size_pop(self, text):
        globalState.taille_pop = text

    def change_fitness_max(self, text):
        globalState.fitness_limit = text

    def change_genome_taille_label(self, text):
        globalState.genome_length = text

    def change_nb_generation(self, text):
        globalState.generation_limit = text

    def on_pushButton_clicked(self):
        dialog = Second(self)
        # self.dialogs.append(dialog)
        # dialog.show()

    def setMutationFLip(self, s):
        globalState.mutation_params = [s, 0.5]


def main():
    app = QApplication(sys.argv)
    main = First()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
