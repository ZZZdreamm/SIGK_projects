import argparse
import sys
from collections import namedtuple
from enum import Enum

import moderngl_window

from phong_window import PhongWindow
from neural_window import NeuralPhongWindow

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../../data/dataset/",
        ],
        PhongWindow
    )
    NEURAL = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
        ],
        NeuralPhongWindow
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('task', nargs='?', default='phong', choices=[t.name.lower() for t in TaskType])
    known, extra = parser.parse_known_args()
    task = TaskType[known.task.upper()]
    moderngl_window.run_window_config(task.window_cls, args=task.window_args + extra)
