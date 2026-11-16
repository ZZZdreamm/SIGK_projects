"""Neural renderer connected with the provided OpenGL renderer.

For every frame a random scene is sampled (same sampler as the dataset), the
reference is rendered with the Phong shader into an off-screen framebuffer and
the same parameters are fed to the trained generator. The window shows
[reference | neural renderer | FLIP error map] and - with --output_path -
saves the panels and the metrics.

    cd SIGK___Projekt_3/src
    uv run python main.py neural --checkpoint ../../runs/geometric_gan/best.pt          # interactive
    uv run python main.py neural --window headless --num_images 8 --output_path ../../results/demo
"""
import csv
import json
import os
import sys
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from phong_window import PhongWindow

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nr import scene

PANELS = 3

QUAD_VS = """
#version 330 core
in vec2 in_position;
in vec2 in_texcoord_0;
out vec2 uv;
void main() { uv = in_texcoord_0; gl_Position = vec4(in_position, 0.0, 1.0); }
"""
QUAD_FS = """
#version 330 core
uniform sampler2D tex;
in vec2 uv;
out vec4 f_color;
void main() { f_color = vec4(texture(tex, uv).rgb, 1.0); }
"""


class NeuralPhongWindow(PhongWindow):
    window_size = (scene.IMAGE_SIZE * PANELS * 2, scene.IMAGE_SIZE * 2)
    aspect_ratio = PANELS
    title = 'SIGK 3 - reference | neural renderer | FLIP'
    save_params = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch
        from moderngl_window import geometry
        from nr.encoding import encode
        from nr.metrics import LPIPS, flip_error, hausdorff, ssim
        from nr.models import Generator

        self.torch, self.encode = torch, encode
        self.flip_error, self.ssim, self.hausdorff = flip_error, ssim, hausdorff
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(self.argv.checkpoint, map_location=self.device)
        self.encoding = ckpt['encoding']
        self.G = Generator(CHANNELS(self.encoding), ckpt['base']).to(self.device).eval()
        self.G.load_state_dict(ckpt['G'])
        self.lpips = LPIPS(self.device)

        size = (scene.IMAGE_SIZE, scene.IMAGE_SIZE)
        self.offscreen = self.ctx.framebuffer(color_attachments=[self.ctx.texture(size, 4)],
                                              depth_attachment=self.ctx.depth_texture(size))
        self.quad_prog = self.ctx.program(vertex_shader=QUAD_VS, fragment_shader=QUAD_FS)
        self.quad = geometry.quad_fs().instance(self.quad_prog)
        self.panel_tex = self.ctx.texture((scene.IMAGE_SIZE * PANELS, scene.IMAGE_SIZE), 3)
        self.panel_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.elapsed = None
        self.metrics_writer = None
        if self.output_path:
            self.metrics_file = open(os.path.join(self.output_path, 'metrics.csv'), 'w', newline='')

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--checkpoint', type=str, default=str(ROOT / 'runs' / 'geometric_gan' / 'best.pt'))
        parser.add_argument('--interval', type=float, default=1.5, help='Seconds per scene (interactive mode)')

    @property
    def headless(self):
        return getattr(self.wnd, '_headless', False)

    def new_scene(self):
        params = self.sample_params()
        self.offscreen.use()
        self.render_scene(params)
        ref = np.asarray(self.read_image(self.offscreen), dtype=np.float32) / 255.0

        v = np.array([*params.model_translation, *params.light_position,
                      *params.material_diffuse, params.material_shininess], dtype=np.float32)
        with self.torch.no_grad():
            p = self.torch.from_numpy(v)[None].to(self.device)
            pred = self.G(self.encode(p, self.encoding))
        lp = float(self.lpips(self.torch.from_numpy(ref).permute(2, 0, 1)[None], pred.cpu())[0])
        pred = pred[0].permute(1, 2, 0).cpu().numpy()
        pred = np.round(pred * 255) / 255
        flip_val, flip_map = self.flip_error(ref, pred, return_map=True)
        metrics = {'FLIP': flip_val, 'LPIPS': lp, 'SSIM': self.ssim(ref, pred),
                   'Hausdorff': self.hausdorff(ref, pred)}

        panel = np.concatenate([ref, pred, np.clip(flip_map, 0, 1)], 1)
        panel8 = (panel * 255).round().astype(np.uint8)
        self.panel_tex.write(np.ascontiguousarray(panel8[::-1]).tobytes())
        print(f'scene {self.frame:3d}: ' + ', '.join(f'{k} {v:.4f}' for k, v in metrics.items()))
        if self.output_path:
            Image.fromarray(panel8).save(os.path.join(self.output_path, f'neural_{self.frame:04}.png'))
            if self.metrics_writer is None:
                self.metrics_writer = csv.DictWriter(self.metrics_file, ['frame', 'params', *metrics])
                self.metrics_writer.writeheader()
            self.metrics_writer.writerow({'frame': self.frame, 'params': json.dumps(params.to_dict()), **metrics})
        self.frame += 1

    def on_render(self, time: float, frame_time: float):
        if self.num_images and self.frame >= self.num_images:
            return
        if self.headless or self.elapsed is None or time - self.elapsed > self.argv.interval:
            self.elapsed = time
            self.new_scene()
        self.wnd.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.panel_tex.use(0)
        self.quad.render()
        if self.num_images and self.frame >= self.num_images:
            self.finish()

    def finish(self):
        if self.output_path:
            self.metrics_file.close()
        print(f'shown {self.frame} scenes')
        self.wnd.close()


def CHANNELS(encoding):
    from nr.encoding import CHANNELS as C
    return C[encoding]
