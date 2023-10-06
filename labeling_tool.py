import argparse
import os
import sys
from functools import partial
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from configs import LABELS

TAGS_KEY = "tags"
TEXT_KEY = "name"


def generate_tokens(name: str) -> List:
    """Generate tokens from a given name

    Args:
        name: the name to be tokenized
    Returns:
        List, a list of tokens
    """
    return name.strip().split(" ")


class NERLabelingApp(QMainWindow):
    """This class encapsulates all functionalities to label NER data"""

    def __init__(self, input_file_data: str, output_file_data: str):
        """
        text: the text to extract the tags
        tokens: the splits of the text to be labeled
        tags: a tuple containing the tag name, and the start and end index of the tag name on text
        """
        super().__init__()
        try:
            assert len(LABELS) > 0
        except:
            raise Exception("No labels found in configs.py")
        # Data
        self.output_file_data = output_file_data
        os.makedirs(os.path.dirname(output_file_data), exist_ok=True)
        self.data_list = self._read_data(
            input_file_data
        )  # List[Dict], a list of the data to be labeled, in format {TEXT_KEY: str, TAGS_KEY: List[List]}
        self.selected_tags = []  # List[List], a list of tags selected by the user
        self.current_text_index = 0
        self.current_token_index = 0
        self.current_button_index = 0
        self.current_tokens = None

        # Widgtes
        self.text_view = None  # QTextEdit: the widget containing the current text
        self.token_view = None  # QTextEdit: the widget containing the current token
        self.tags_view = None  # QLabel: the widget containing the tags created
        self.current_index_view = None  # QLabel: the widget containing the current index

        # Buttons
        self.save_button = None
        self.clear_button = None
        self.next_text_button = None
        self.previous_text_button = None
        self.next_token_button = None
        self.previous_token_button = None
        self.label_buttons = []  # List[QPushButton]: the list of buttons for the labels

        # Init
        self._load_widgets()
        self._create_actions()
        self._load_layout()
        self.update_all()

        self.print_shortkeys()

    def update_all(self):
        self._update_text()
        self._update_token()
        self._update_labels()

    def reset_all(self):
        self.selected_tags = []
        self.update_all()

    def _update_text(self):
        """Update the text view"""
        text = self.data_list[self.current_text_index].get(TEXT_KEY)
        self.current_token_index = 0
        self.text_view.setText(text)
        self.current_tokens = generate_tokens(text)
        self.current_index_view.setText(str(self.current_text_index))

    def _update_token(self):
        """Update the tokens view"""
        current_token = self.current_tokens[self.current_token_index]
        self.token_view.setText(current_token)

    def _update_labels(self):
        """Update the labels view"""
        text = ""
        saved_tags = self.data_list[self.current_text_index][TAGS_KEY]
        for tag in saved_tags + self.selected_tags:
            tag_name, start, end = tag
            token = self.text_view.text()[start:end]
            text += f"{tag_name} : {token}\n"
        self.tags_view.setText(text)

    def save_data_action(self):
        """Save the data to the file"""
        self.data_list[self.current_text_index][TAGS_KEY] += self.selected_tags
        self.selected_tags = []
        self._save_to_file(self.output_file_data)

    def clear_action(self):
        """Clear the data to the current text"""
        self.data_list[self.current_text_index][TAGS_KEY] = []
        self.reset_all()

    def previous_text_action(self):
        self.save_data_action()  # Avoid mistakes of don't save the current data
        self.current_text_index -= 1
        self.current_text_index = max(self.current_text_index, 0)
        self.reset_all()

    def next_text_action(self):
        self.save_data_action()
        self.current_text_index += 1
        self.current_text_index = min(self.current_text_index, len(self.data_list) - 1)
        self.reset_all()

    def previous_token_action(self):
        self.current_token_index -= 1
        self.current_token_index = max(self.current_token_index, 0)
        self._update_token()

    def next_token_action(self):
        self.current_token_index += 1
        self.current_token_index = min(self.current_token_index, len(self.current_tokens) - 1)
        self._update_token()

    def label_button_clicked(self, id: int):
        """Handle the click of a label button"""
        # Get the start and end index of the token
        start = 0
        for idx in range(self.current_token_index):
            curr_token = self.current_tokens[idx]
            start += len(curr_token) + 1  # +1 for the space
        token = self.current_tokens[self.current_token_index]
        end = start + len(token)

        updated = False
        label = LABELS[id]
        saved_tags = self.data_list[self.current_text_index][TAGS_KEY]

        # Check if the token is already labeled and update it
        for tag in self.selected_tags + saved_tags:
            if tag[1] == start and tag[2] == end:
                tag[0] = label
                updated = True
                break

        if not updated:
            tag = [label, start, end]
            if tag not in self.selected_tags:
                self.selected_tags.append(tag)

        self.next_token_action()
        self._update_labels()

    def next_button_action(self):
        self.current_button_index = (self.current_button_index + 1) % len(self.label_buttons)
        self._update_selected_button_color()

    def previous_button_action(self):
        self.current_button_index = (self.current_button_index - 1) % len(self.label_buttons)
        self._update_selected_button_color()

    def _update_selected_button_color(self):
        for button in self.label_buttons:
            button.setStyleSheet("")  # Reset style for all buttons
        self.label_buttons[self.current_button_index].setStyleSheet("background-color: yellow;")

    def click_tag_button_action(self):
        self.label_button_clicked(self.current_button_index)

    def _save_to_file(self, file_data: str):
        """Save the data to the file"""
        with open(file_data, "w+") as f:
            f.write("[")
            for elem in self.data_list:
                f.write(f"{repr(elem)},\n")
            f.write("]\n")

    def _read_data(self, file_data: str) -> List:
        """Read the data from the file

        Args:
            file_data: the file containing the data to be labeled
        Returns:
            List, a list of the data
        """
        with open(file_data, "r") as f:
            # -1 to remove the \n
            names = [name[:-1] for name in f.readlines()]
        data = [{TEXT_KEY: name, TAGS_KEY: []} for name in names]
        return data

    def print_shortkeys(self):
        text = f"""Shortkeys:
        a: previous tag
        d: next tag
        w: click selected button
        s: save
        c: clear
        left arrow: previous token
        right arrow: next token
        """
        print(text)

    def _load_widgets(self):
        self.text_view = QLabel(self)
        self.text_view.setFixedHeight(self.text_view.fontMetrics().height() + 10)
        self.text_view.setAlignment(Qt.AlignCenter)

        self.token_view = QLabel(self)
        self.token_view.setFixedHeight(self.token_view.fontMetrics().height() + 10)
        self.token_view.setAlignment(Qt.AlignCenter)
        self.token_view.setStyleSheet("background-color: lightgray;")

        self.tags_view = QLabel(self)
        self.tags_view.setStyleSheet("background-color: lightgray;")

        # Buttons
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_data_action)
        self.save_button.setStyleSheet(":hover {background-color: lightgreen;}")

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_action)
        self.clear_button.setStyleSheet(":hover {background-color: red;}")

        self.next_text_button = QPushButton(">>", self)
        self.next_text_button.clicked.connect(self.next_text_action)

        self.previous_text_button = QPushButton("<<", self)
        self.previous_text_button.clicked.connect(self.previous_text_action)

        self.next_token_button = QPushButton(">", self)
        self.next_token_button.clicked.connect(self.next_token_action)

        self.previous_token_button = QPushButton("<", self)
        self.previous_token_button.clicked.connect(self.previous_token_action)

        self.current_index_view = QLabel(self)
        self.current_index_view.setText("0")
        self.current_index_view.setAlignment(Qt.AlignCenter)

        for id, label in enumerate(LABELS):
            button = QPushButton(label, self)
            button.clicked.connect(partial(self.label_button_clicked, id))
            self.label_buttons.append(button)

    def _load_layout(self):
        self.setWindowTitle("NER Labeling Interface")
        self.resize(500, 300)

        # The labels handler (Save, Clear)
        labels_handler_layout = QHBoxLayout()
        labels_handler_layout.addWidget(self.save_button)
        labels_handler_layout.addWidget(self.clear_button)

        labels_tools_widget = QWidget()
        labels_tools_widget.setLayout(labels_handler_layout)

        # Tags Buttons
        tag_buttons_intermediate_layout = QHBoxLayout()
        tag_buttons_layout = QHBoxLayout()
        intermediate_buttons = [x for x in self.label_buttons if x.text().startswith("I-")]
        for button in self.label_buttons:
            # Split the buttons in two lines, one for the intermediate tags, and one for the others
            if button in intermediate_buttons:
                tag_buttons_intermediate_layout.addWidget(button)
            else:
                tag_buttons_layout.addWidget(button)

        tags_widget = QWidget()
        tags_widget.setLayout(tag_buttons_layout)
        tags_intermediate_widget = QWidget()
        tags_intermediate_widget.setLayout(tag_buttons_intermediate_layout)
        # The text handler (<<, <, index, >, >>)
        text_handler_layout = QHBoxLayout()
        text_handler_layout.addWidget(self.previous_text_button)
        text_handler_layout.addWidget(self.previous_token_button)
        text_handler_layout.addWidget(self.current_index_view)
        text_handler_layout.addWidget(self.next_token_button)
        text_handler_layout.addWidget(self.next_text_button)

        text_tools_widget = QWidget()
        text_tools_widget.setLayout(text_handler_layout)

        # Main Layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_view)
        layout.addWidget(self.token_view)
        layout.addWidget(tags_widget)
        layout.addWidget(tags_intermediate_widget)
        layout.addWidget(self.tags_view)
        layout.addWidget(labels_tools_widget)
        layout.addWidget(text_tools_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _create_actions(self):
        # Create the action
        go_to_next_text = QAction("Next Text", self)
        go_to_previous_text = QAction("Previous Text", self)
        save_action = QAction("Save", self)
        clear_action = QAction("Clear", self)
        next_button = QAction("Button next", self)
        previous_button = QAction("Button previous", self)
        click_button = QAction("Click Button", self)

        # Set the shortcut
        go_to_next_text.setShortcut(QKeySequence(Qt.Key_Right))
        go_to_previous_text.setShortcut(QKeySequence(Qt.Key_Left))
        save_action.setShortcut(QKeySequence(Qt.Key_S))
        clear_action.setShortcut(QKeySequence(Qt.Key_C))
        next_button.setShortcut(QKeySequence(Qt.Key_D))
        previous_button.setShortcut(QKeySequence(Qt.Key_A))
        click_button.setShortcut(QKeySequence(Qt.Key_W))

        # Connect the action to the function
        go_to_next_text.triggered.connect(self.next_text_action)
        go_to_previous_text.triggered.connect(self.previous_text_action)
        save_action.triggered.connect(self.save_data_action)
        clear_action.triggered.connect(self.clear_action)
        next_button.triggered.connect(self.next_button_action)
        previous_button.triggered.connect(self.previous_button_action)
        click_button.triggered.connect(self.click_tag_button_action)

        # Add the action to the window
        self.addAction(go_to_next_text)
        self.addAction(go_to_previous_text)
        self.addAction(save_action)
        self.addAction(clear_action)
        self.addAction(next_button)
        self.addAction(previous_button)
        self.addAction(click_button)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    arguments = parser.parse_args()

    app = QApplication(sys.argv)
    window = NERLabelingApp(arguments.input_file, arguments.output_file)
    window.show()
    sys.exit(app.exec_())
