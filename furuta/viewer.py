import time
from abc import abstractmethod

import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

from furuta.logger import SimpleLogger
from furuta.utils import ALPHA, THETA


class AbstractViewer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def close(self):
        pass


class Viewer3D(AbstractViewer):
    def __init__(self, robot: pin.RobotWrapper = None):
        self.robot = robot
        self.viewer = Viewer()
        # Attach the robot to the viewer scene
        self.robot.setVisualizer(Panda3dVisualizer())
        self.robot.initViewer(viewer=self.viewer)
        self.robot.loadViewerModel(group_name=self.robot.model.name)

    def display(self, q: np.ndarray):
        self.robot.display(q)
        return self.viewer.get_screenshot(requested_format="RGB")

    def close(self):
        self.viewer.close()

    def animate(self, times: np.ndarray, states: np.ndarray):
        # Initial state
        q = np.array(states)[:, :2]
        self.display(q[0])
        time.sleep(1.0)
        tic = time.time()
        for i in range(1, len(times)):
            toc = time.time()
            time.sleep(max(0, times[i] - times[i - 1] - (toc - tic)))
            tic = time.time()
            if i % 10 == 0:
                self.display(q[i])

    def animate_log(self, log: SimpleLogger):
        self.animate(log.times, log.states)


class Viewer2D(AbstractViewer):
    def __init__(self, render_fps: int, render_mode: str = None):
        self.render_fps = render_fps
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None

    def display(self, state):
        # https://github.com/Farama-Foundation/Gymnasium/blob/6baf8708bfb08e37ce3027b529193169eaa230fd/gymnasium/envs/classic_control/cartpole.py#L229
        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            else:
                print(f"invalid render mode : {self.render_mode}")
                return

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if state is None:
            return None

        world_width = 2 * np.pi
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * 0.5)  # TODO use the right pole len?
        cartwidth = 50.0
        cartheight = 30.0

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0

        # make sure theta stays between 0 and 2 * pi
        theta = (state[THETA] + np.pi) % (2 * np.pi)
        cartx = theta * scale
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(state[ALPHA] + np.pi)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
