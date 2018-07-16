import vispy
import vispy.color
import vispy.scene
import vispy.app
import vispy.visuals
import numpy as np
import os
import TurntableCamera as tc


class FireMap(vispy.color.colormap.BaseColormap):
    colors = [(1.0, 1.0, 1.0, 0.0),
              (1.0, 1.0, 0.0, 0.05),
              (1.0, 0.0, 0.0, 0.1)]

    glsl_map = """
    vec4 fire(float t) {
        return mix(mix($color_0, $color_1, t),
                   mix($color_1, $color_2, t*t), t);
    }
    """


def show():
    vispy.app.run()


def plot3d(data, colormap = FireMap(), view=None):
    VolumePlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.VolumeVisual)
    # Add a ViewBox to let the user zoom/rotate
    # build canvas
    if view is None:
        canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
        view = canvas.central_widget.add_view(camera=tc.TurntableCamera())
        view.camera = 'turntable'
        view.camera.fov = 0
        view.camera.distance = 7200
        view.camera.elevation = 31
        view.camera.azimuth = 0
        view.camera.depth_value = 100000000
        cc = (np.array(data.shape) // 2)
        view.camera.center = cc

    return VolumePlot3D(data.transpose([2, 1, 0]), method='translucent', relative_step_size=1.5,
                        parent=view.scene, cmap=colormap)


if __name__ == '__init__':

    savedir = '/mnt/raid/UnsupSegment/patches'

    image = np.load(os.path.join(savedir, '10-43-24_IgG_UltraII[04 x 08]_C00_patch_760_520_320.npy'))
    plot3d(image)
    show()