import time
from abc import ABC, abstractmethod

import numpy as np
import pinocchio as pin
from panda3d_viewer import Viewer
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer

from furuta.logger import SimpleLogger
from furuta.utils import ALPHA, THETA


class AbstractViewer(ABC):
    @classmethod
    @abstractmethod
    def __init__(cls):
        pass

    @classmethod
    @abstractmethod
    def display(cls, state: np.ndarray) -> (np.ndarray | None):
        pass

    @classmethod
    @abstractmethod
    def close(cls):
        pass


class Viewer3D(AbstractViewer):
    def __init__(cls, robot: pin.RobotWrapper = None):
        cls.robot = robot
        cls.viewer = Viewer()
        # Attach the robot to the viewer scene
        cls.robot.setVisualizer(Panda3dVisualizer())
        cls.robot.initViewer(viewer=cls.viewer)
        cls.robot.loadViewerModel(group_name=cls.robot.model.name)

    def display(cls, state: np.ndarray) -> np.ndarray:
        cls.robot.display(state)
        return cls.viewer.get_screenshot(requested_format="RGB")

    def close(cls):
        cls.viewer.close()

    def animate(cls, times: np.ndarray, states: np.ndarray):
        # Initial state
        q = np.array(states)[:, :2]
        cls.display(q[0])
        time.sleep(1.0)
        tic = time.time()
        for i in range(1, len(times)):
            toc = time.time()
            time.sleep(max(0, times[i] - times[i - 1] - (toc - tic)))
            tic = time.time()
            if i % 10 == 0:
                cls.display(q[i])

    def animate_log(cls, log: SimpleLogger):
        cls.animate(log.times, log.states)


class Viewer2D(AbstractViewer):
    def __init__(cls, render_fps: int = 50, render_mode: str = None):
        cls.render_fps = render_fps
        cls.render_mode = render_mode

        cls.screen_width = 600
        cls.screen_height = 400
        cls.screen = None
        cls.clock = None

    def display(cls, state: np.ndarray):
        # https://github.com/Farama-Foundation/Gymnasium/blob/6baf8708bfb08e37ce3027b529193169eaa230fd/gymnasium/envs/classic_control/cartpole.py#L229
        import pygame
        from pygame import gfxdraw

        if cls.screen is None:
            pygame.init()
            if cls.render_mode == "human":
                pygame.display.init()
                cls.screen = pygame.display.set_mode((cls.screen_width, cls.screen_height))
            elif cls.render_mode == "rgb_array":
                cls.screen = pygame.Surface((cls.screen_width, cls.screen_height))
            else:
                print(f"invalid render mode : {cls.render_mode}")
                return

        if cls.clock is None:
            cls.clock = pygame.time.Clock()

        if state is None:
            return None

        world_width = 2 * np.pi
        scale = cls.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * 0.5)  # TODO use the right pole len?
        cartwidth = 50.0
        cartheight = 30.0

        cls.surf = pygame.Surface((cls.screen_width, cls.screen_height))
        cls.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0

        # make sure theta stays between 0 and 2 * pi
        theta = (state[THETA] + np.pi) % (2 * np.pi)
        cartx = theta * scale
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(cls.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(cls.surf, cart_coords, (0, 0, 0))

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
        gfxdraw.aapolygon(cls.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(cls.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            cls.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            cls.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(cls.surf, 0, cls.screen_width, carty, (0, 0, 0))

        cls.surf = pygame.transform.flip(cls.surf, False, True)
        cls.screen.blit(cls.surf, (0, 0))
        if cls.render_mode == "human":
            pygame.event.pump()
            cls.clock.tick(cls.render_fps)
            pygame.display.flip()

        elif cls.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(cls.screen)), axes=(1, 0, 2))

    def close(cls):
        if cls.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
