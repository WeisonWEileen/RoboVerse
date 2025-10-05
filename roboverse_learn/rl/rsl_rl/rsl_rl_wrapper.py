"""Wrapper for rsl_rl 2.3.10, align OnPolicyRunnner in rsl_rl with metasim."""

from __future__ import annotations

import torch
from loguru import logger as log
from rsl_rl.env import VecEnv

from metasim.scenario.scenario import ScenarioCfg
from metasim.constants import SimType

from metasim.utils.setup_util import get_sim_handler_class
from metasim.utils.state import list_state_to_tensor

from metasim.sim.queries.net_contact_force import NetContactForce

class RslRlWrapper(VecEnv):
    """
    Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__()


        if SimType(scenario.simulator) not in [SimType.ISAACSIM, SimType.ISAACGYM,SimType.ISAACLAB, SimType.GENESIS]:
            raise NotImplementedError(
                f"RslRlWrapper in Roboverse now only supports {SimType.ISAACGYM}, but got {scenario.sim}"
            )
        self.device = torch.device(scenario.device if torch.cuda.is_available() else "cpu")
        log.info(f"using device {self.device}")
        self._parse_cfg(scenario)

        # load simulator handler
        env_class = get_sim_handler_class(SimType(scenario.simulator))
        # TODO fixme
        # contact_force_queries = {"net_contact_force": NetContactForce()}
        # env = env_class(scenario_cfg=scenario, optional_queries=contact_force_queries)
        env = env_class(scenario_cfg=scenario)
        env.launch()

        self.env = env
        self._get_init_states(scenario)
        env.set_states(self.init_states)
        self.use_vision = scenario.task.use_vision

        if self.use_vision:
            self.vision_height = scenario.task.cameras[0].height
            self.vision_width = scenario.task.cameras[0].width


    def _init_buffers(self):
        """Initialize buffers for rsl_rl compatibility."""
        # Initialize observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float32)
        if self.use_vision:
            self.vision_buf = torch.zeros((self.num_envs, 3, self.vision_height, self.vision_width), device=self.device, dtype=torch.float32)
        if self.use_vision:
            self.extra_buf = {
            "observations": {
                "critic": (self.privileged_obs_buf, self.vision_buf),  # For PPO training
                # Add other observation types as needed
            }}
        else:
            self.extra_buf = {
            "observations": {
                "critic": self.privileged_obs_buf,  # For PPO training
                # Add other observation types as needed
            }
            # TODO add episodic info and log
        }
        self._episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


    def _parse_cfg(self, scenario: ScenarioCfg):
        # loading task-specific configuration
        self.scenario = scenario
        self.robot = scenario.robots[0]
        self.num_envs = scenario.num_envs
        self.num_obs = scenario.task.num_observations
        self.num_actions = scenario.robots[0].num_joints
        self.num_privileged_obs = scenario.task.num_privileged_obs
        self.max_episode_length = scenario.task.max_episode_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = scenario.task
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(scenario.task.ppo_cfg)


    def _get_init_states(self, scenario):
        """ Get initial states from the scenario configuration."""

        init_states_list = getattr(scenario.task, 'init_states', None)
        if init_states_list is None:
            raise AttributeError(f"'task cfg' has no attribute 'init_states', please add it in your scenario config!")

        if len(init_states_list) < self.num_envs:
            init_states_list = (
                init_states_list * (self.num_envs // len(init_states_list))
                + init_states_list[: self.num_envs % len(init_states_list)]
            )
        else:
            init_states_list = init_states_list[: self.num_envs]

        self.init_states = init_states_list

        if scenario.simulator == "isaacsim":
            #tensorize the initial states as TensorState, now we only support IsaacGym
            self.init_states = list_state_to_tensor(self.env, init_states_list, device=self.device)


    def get_observations(self):
        """design from config"""
        return self.obs_buf, self.extra_buf

    def get_privileged_observations(self):
        """design from config"""
        return self.privileged_obs_buf

    def step(self, actions):
        raise NotImplementedError

    def get_visual_observations(self):
        raise NotImplementedError

    def reset(self, env_ids=None):
        """
        Reset state in the env and buffer in the wrapper
        """
        if env_ids is None:
            env_ids = list(range(self.env.num_envs))

        # reset in the env
        self.env.reset(env_ids)
