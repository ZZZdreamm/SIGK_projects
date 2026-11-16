import json
import os.path
import sys
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nr import scene


class PhongWindow(BaseWindow):
    """Provided Phong renderer + random scene sampling for dataset generation.

    Every frame draws a new random scene (object position, diffuse colour,
    shininess, light position), renders it and - with --output_path - saves
    `image_XXXX.png` and the parameters to `params.jsonl`. The window closes
    after --num_images frames (0 = run forever, interactive preview).
    """
    save_params = True

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.rng = np.random.default_rng(self.argv.seed)
        self.num_images = self.argv.num_images
        self.reject_stats = {}
        self.params_file = None
        if self.output_path and self.save_params:
            self.params_file = open(os.path.join(self.output_path, 'params.jsonl'), 'w')

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--num_images', type=int, default=0, help='Frames to render (0 = infinite)')
        parser.add_argument('--seed', type=int, default=0, help='Seed of the scene sampler')
        parser.add_argument('--unconstrained', action='store_true',
                            help='Do not reject degenerate scenes (raw U[-20,20] sampling)')

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def sample_params(self) -> scene.SceneParams:
        return scene.sample_scene(self.rng, constrained=not self.argv.unconstrained,
                                  stats=self.reject_stats)

    def render_scene(self, params: scene.SceneParams):
        """Draw one scene with the Phong shader into the currently bound framebuffer."""
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        model_translation = params.model_translation
        material_diffuse = np.array(params.material_diffuse) / 255.0
        material_shininess = params.material_shininess
        light_position = params.light_position

        camera_position = scene.CAMERA_POSITION.tolist()
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(scene.FOV_Y_DEG, 1.0, scene.NEAR, scene.FAR)
        lookat = Matrix44.look_at(
            camera_position,
            scene.CAMERA_TARGET.tolist(),
            scene.CAMERA_UP.tolist(),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()

    def read_image(self, fbo) -> Image.Image:
        return (
            Image.frombuffer('RGBA', fbo.size, fbo.read(components=4))
                 .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                 .convert('RGB')
        )

    def on_render(self, time: float, frame_time: float):
        if self.num_images and self.frame >= self.num_images:
            return
        params = self.sample_params()
        self.render_scene(params)
        if self.output_path:
            img = self.read_image(self.wnd.fbo)
            name = f'image_{self.frame:04}.png'
            img.save(os.path.join(self.output_path, name))
            self.params_file.write(json.dumps({'file': name, **params.to_dict()}) + '\n')
        self.frame += 1
        if self.num_images and self.frame >= self.num_images:
            self.finish()

    def finish(self):
        if self.params_file:
            self.params_file.close()
        if self.params_file and self.save_params:
            with open(os.path.join(self.output_path, 'sampling_stats.json'), 'w') as f:
                json.dump(self.reject_stats, f, indent=2)
        print(f'rendered {self.frame} images, rejection stats: {self.reject_stats}')
        self.wnd.close()
