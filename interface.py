import sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QWidget,
    QLineEdit,
)
import Lanceur
import math
import random


class GlobalParameter:
    def __init__(
        self,
        seed,
        taille_pop,
        mutation_params,
        selector_operator,
        selection_params,
        fitness_limit,
        generation_limit,
        genome_length,
        # sudoku_size,
        nb_run,
        croisement_param,
        selected_problem,
    ):
        self.seed = seed
        self.selector_operator = selector_operator
        self.mutation_params = mutation_params
        self.selection_params = selection_params
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit
        # self.sudoku_size = sudoku_size
        self.genome_length = genome_length
        self.taille_pop = taille_pop
        self.nb_run = nb_run
        self.croisement_param = croisement_param
        self.selected_problem = selected_problem


# 1-flip etc... AOS_UCB AOS_PM
global_state = GlobalParameter(
    random.randint(0, 10000),
    10,
    ["1-flip", 0.5],
    "1-flip",
    "selection_tournois_parmi_s_randoms",
    1000,
    1000,
    484,
    # 4,
    10,
    "uniform_crossover",
    "OneMax",
)


class Second(QMainWindow):

    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        Lanceur.launch_the_launcher(global_state)

        class MainWindow(QMainWindow):

            def __init__(self):
                super(MainWindow, self).__init__()
                self.title = "Algo génétique"
                self.setWindowTitle(self.title)

                label = QLabel(self)
                pixmap = QPixmap("debug/plot.png")
                label.setPixmap(pixmap)
                self.setCentralWidget(label)
                self.resize(pixmap.width(), pixmap.height())

        self.w = MainWindow()
        self.w.resize(600, 600)
        self.w.show()


def change_size_pop(text):
    global_state.taille_pop = text


def change_nb_run(text):
    global_state.nb_run = text


class First(QMainWindow):

    def __init__(self, parent=None):
        super(First, self).__init__(parent)
        self.setGeometry(500, 500, 500, 250)
        self.title = "Menu de sélection"
        self.setWindowTitle(self.title)

        self.buttonRun = QPushButton("Run")
        self.cleanupButton = QPushButton("Cleanup the plot")
        self.fitnessLabel = QLabel("Validation")
        self.choixFitness = QLabel(
            "Choix fitness par encore disponible - onemax par défaut"
        )

        self.validationLabel = QLabel("Validation")

        self.selectionLabel = QLabel("Sélection")
        self.selectionChoix = QComboBox()
        self.selectionChoix.addItem("selection_tournois_parmi_s_randoms")
        self.selectionChoix.addItem("selection_pair_better")
        self.selectionChoix.addItem("selection_pair")

        self.mutationLabel = QLabel("Mutation")

        self.mutationChoix = QComboBox()
        self.mutationChoix.addItem("1-flip")
        self.mutationChoix.addItem("2-flip")
        self.mutationChoix.addItem("3-flip")
        self.mutationChoix.addItem("4-flip")
        self.mutationChoix.addItem("5-flip")
        self.mutationChoix.addItem("bitflip")
        self.mutationChoix.addItem("AOS_UCB")
        self.mutationChoix.addItem("AOS_PM")
        self.mutationChoix.addItem("OS_RANDOM_AVEUGLE")
        self.mutationChoix.addItem("OS_MANUAL")

        self.croisementLabel = QLabel("Croisement")

        self.croisementChoix = QComboBox()
        self.croisementChoix.addItem("uniform_crossover")
        self.croisementChoix.addItem("single_point_crossover")

        self.problemLabel = QLabel("Problème à traiter")

        self.selectedProblem = QComboBox()
        self.selectedProblem.addItem("OneMax")
        self.selectedProblem.addItem("Sudoku")
        self.selectedProblem.addItem("N-Reine")
        self.selectedProblem.addItem("KnapSack - WIP")
        self.selectedProblem.addItem("Quadratic knapsack - WIP")
        self.selectedProblem.addItem("PPP - WIP")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.fitnessLabel)
        self.layout.addWidget(self.choixFitness)
        self.layout.addWidget(self.validationLabel)

        self.layout.addWidget(self.buttonRun)
        self.layout.addWidget(self.selectionLabel)
        self.layout.addWidget(self.selectionChoix)
        self.layout.addWidget(self.mutationLabel)
        self.layout.addWidget(self.mutationChoix)
        self.layout.addWidget(self.croisementLabel)
        self.layout.addWidget(self.croisementChoix)
        self.layout.addWidget(self.problemLabel)
        self.layout.addWidget(self.selectedProblem)
        self.layout.addWidget(self.cleanupButton)

        self.sizePopLabel = QLabel("nb d'individus (conseillé - minimum 4)")
        self.sizePop = QLineEdit()
        self.sizePop.setText("10")
        self.layout.addWidget(self.sizePopLabel)
        self.layout.addWidget(self.sizePop)
        self.sizePop.textChanged.connect(change_size_pop)

        self.nbRunLabel = QLabel("Nombre de run")
        self.nbRun = QLineEdit()
        self.nbRun.setText("10")
        self.layout.addWidget(self.nbRunLabel)
        self.layout.addWidget(self.nbRun)
        self.nbRun.textChanged.connect(change_nb_run)

        self.fitnessMaxLabel = QLabel("Fitness maximum - (stop if reached)")
        self.fitnessMax = QLineEdit()
        self.fitnessMax.setText("20000")
        self.layout.addWidget(self.fitnessMaxLabel)
        self.layout.addWidget(self.fitnessMax)
        self.fitnessMax.textChanged.connect(self.change_fitness_max)

        # OneMax
        self.genomeTailleLabel = QLabel("genome taille")
        self.genomeTaille = QLineEdit()
        self.genomeTaille.setText("484")
        self.layout.addWidget(self.genomeTailleLabel)
        self.layout.addWidget(self.genomeTaille)
        self.genomeTaille.textChanged.connect(self.change_genome_taille_label)

        # N-Reine
        self.nreineTailleLabel = QLabel("Taille N-Reine (nb de reine) ")
        self.nreineTaille = QLineEdit()
        self.nreineTaille.setText("5")
        self.layout.addWidget(self.nreineTailleLabel)
        self.layout.addWidget(self.nreineTaille)
        self.nreineTaille.textChanged.connect(self.change_nreine_taille_label)

        self.nreineTailleLabel.hide()
        self.nreineTaille.hide()

        # Sudoku
        self.sudokuTailleLabel = QLabel("Taille Sudoku (doit être un carré parfait) ")
        self.sudokuTaille = QLineEdit()
        self.sudokuTaille.setText("5")
        self.layout.addWidget(self.sudokuTailleLabel)
        self.layout.addWidget(self.sudokuTaille)
        self.sudokuTaille.textChanged.connect(self.change_sudoku_taille_label)

        self.sudokuTailleLabel.hide()
        self.sudokuTaille.hide()

        self.generationNbLabel = QLabel("Itération / génération")
        self.generationNb = QLineEdit()
        self.generationNb.setText("1000")
        self.layout.addWidget(self.generationNbLabel)
        self.layout.addWidget(self.generationNb)
        self.generationNb.textChanged.connect(self.change_nb_generation)
        self.dialogs = ""
        self.buttonRun.clicked.connect(self.on_pushButton_clicked)
        self.cleanupButton.clicked.connect(self.on_cleanup_button_clicked)
        self.mutationChoix.currentTextChanged.connect(self.setMutationFlip)
        self.croisementChoix.currentTextChanged.connect(self.setCroisementChoix)
        self.selectedProblem.currentTextChanged.connect(self.setSelectedProblem)
        self.selectionChoix.currentTextChanged.connect(self.setSelectionChoix)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.selectedProblem.currentIndexChanged.connect(
            self.update_genome_taille_visibility
        )

        self.show()

    def update_genome_taille_visibility(self):
        self.genomeTailleLabel.hide()
        self.genomeTaille.hide()

        self.nreineTailleLabel.hide()
        self.nreineTaille.hide()

        self.sudokuTailleLabel.hide()
        self.sudokuTaille.hide()

        if self.selectedProblem.currentText() == "OneMax":
            self.genomeTailleLabel.show()
            self.genomeTaille.show()
        if self.selectedProblem.currentText() == "N-Reine":
            self.nreineTailleLabel.show()
            self.nreineTaille.show()
        if self.selectedProblem.currentText() == "Sudoku":
            self.sudokuTailleLabel.show()
            self.sudokuTaille.show()
            self.fitnessMax.setText("0")

    def change_fitness_max(self, text):
        global_state.fitness_limit = text

    def change_genome_taille_label(self, text):
        global_state.genome_length = text

    # URG: the modification is not taken into account directly, fix it
    def change_nreine_taille_label(self, text):
        global_state.genome_length = int(text) ** 2  # Autosizer

    # URG: the modification is not taken into account directly, fix it
    def change_sudoku_taille_label(self, text):
        size = int(text)
        global_state.genome_length = (
            size * size * math.ceil(math.log2(size))
        )  # Autosizer
        sudoku_size = size

    def change_nb_generation(self, text):
        global_state.generation_limit = text

    def on_pushButton_clicked(self):
        dialog = Second(self)

    def on_cleanup_button_clicked(self):
        Lanceur.cleanup_graph()

    def setMutationFlip(self, s):
        global_state.mutation_params = [s, 0.5]
        global_state.selector_operator = s

    def setSelectionChoix(self, s):
        global_state.selection_params = s

    def setCroisementChoix(self, s):
        global_state.croisement_param = s

    def setSelectedProblem(self, s):
        global_state.selected_problem = s


def main():
    app = QApplication(sys.argv)
    main = First()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
