from calendar import different_locale
from collections import deque

import gym
from gym import spaces

import numpy as np


class QuadHover(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ref_div=1.0,
        delay=3,
        noise=0.1,
        noise_p=0.1,
        g=9.81,
        g_bounds=(-0.8, 0.5),
        noise_a=0.0,
        dead_a=0.0,
        thrust_tc=0.02,
        settle=1.0,
        wind=0.1,
        h0=5.0,
        h_blind=5.0,
        dt=0.02,
        ds_act=1,
        jitter=0.0,
        min_h=0.05,
        max_h=10.0, #changed
        max_t=30.0,
        seed=0,
    ):
        # Keywords
        self.G = g
        self.dt = dt
        self.ds_act = ds_act  # action selection every ds_act steps (so 1 means each step)
        self.jitter_prob = jitter  # probability of computational jitter
        self.min_h = min_h
        self.max_h = max_h
        self.max_t = max_t
        self.settle = settle  # initial settling period without any control
        self.delay = delay  # delay in steps
        self.noise_std = noise  # white noise
        self.noise_p_std = noise_p  # noise proportional to divergence
        self.noise_a_std = noise_a  # noise (in g's) added to action
        self.deadband_a = dead_a  # deadband (in g's) for action
        self.wind_std = wind
        self.thrust_tc = thrust_tc  # thrust time constant
        self.h_blind = h_blind  # height above which divergence can't be observed

#####
        self.ref_div = ref_div
#####


        # new
        self.reward = 0 
        self.height_reward = h0
        #

        # Seed
        self.seed(seed)

        # Reset to get initial observation
        self.reset(h0)

        # Initialize spaces
        # Thrust value as action, (div, div_dot) as observation
        self.action_space = spaces.Box(low=g_bounds[0], high=g_bounds[1], shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Check values
        self.checks()

    def checks(self):
        # Check values
        assert self.delay >= 0 and isinstance(self.delay, int)
        assert self.noise_std >= 0.0 and self.noise_p_std >= 0.0
        assert self.noise_a_std >= 0.0
        assert self.deadband_a >= 0.0
        assert (self.action_space.low[0] >= -1.0 and self.G == 9.81) or (self.action_space.low[0] >= -9.81 and self.G == 1.0)
        assert self.thrust_tc > 0.0
        assert self.settle >= 0.0
        assert self.wind_std >= 0.0
        assert self.jitter_prob >= 0.0 and self.jitter_prob <= 1.0
        assert self.dt > 0.0
        assert self.ds_act > 0 and isinstance(self.ds_act, int)
        assert self.min_h >= 0.0
        assert self.max_h > self.min_h
        assert self.max_t > 0.0
        assert (self.delay + 1) * self.dt < self.settle
        assert self.h_blind >= self.min_h and self.h_blind <= self.max_h

    def step(self, action, jitter_prob=None):
        # Set computational jitter
        if jitter_prob is None:
            jitter_prob = self.jitter_prob

        # Update wind
        self._get_wind()

        # Take action
        self._get_action(action)

        # Update state with forward Euler
        self.state += (
            self._get_state_dot()
            + [0.0, self.wind + self.disturbance[0], self.disturbance[1]]
        ) * self.dt   #integration
        self.t += self.dt
        self.steps += 1


        # Get reward
        reward = self._get_reward()

        self.reward += reward

        # Check whether done
        done = self._check_done() or self._check_done_cost_function()

        # Clamp state to prevent negative altitudes and too high max time
        # And prevent another step due to jitter
        if done:
            #add total reward calculations here
            

            self.state[0] = self._clamp(self.state[0], self.min_h, self.max_h)
            self.t = self._clamp(self.t, 0.0, self.max_t)
            jitter_prob = 0.0


            if self.state[0] == self.max_h :
                # self.reward = self.reward + np.abs((self.max_t - self.t) * self.max_h)

                t_adm = self.t
                while self.height_reward >= self.min_h:
                    self.height_reward = self.height_reward - self.height_reward/2*self.dt
                    self.reward = self.reward + np.abs(self.dt * (self.max_h- self.height_reward))

                # self.reward = self.reward + np.abs((t_adm - self.t) * self.max_h)
            if np.abs(self.state[0] - self.min_h)<0.01:
                t_adm = self.t
                while self.height_reward >= self.min_h:
                    self.height_reward = self.height_reward - self.height_reward/2*self.dt
                    # self.reward = self.reward + np.abs(self.height_reward/(self.height_reward/4) *self.height_reward *0.5)#dit is nog fout
                    self.reward = self.reward + self.dt*self.height_reward



        

        # Computational jitter: do another step (without possibility of another delay)
        if np.random.random() < jitter_prob:
            self._get_obs()
            return self.step(action, jitter_prob=0.0)

        return self._get_obs(), reward, done, {}, self.height_reward

    def set_disturbance(self, v_disturbance, a_disturbance):
        self.disturbance = [v_disturbance, a_disturbance]

    def unset_disturbance(self):
        self.disturbance = [0.0, 0.0]

    def _get_action(self, action):
        # Take new action
        if not self.steps % self.ds_act:
            self.action = action + np.random.normal(0.0, self.noise_a_std)
            # Deadband
            if abs(self.action) < self.deadband_a:
                self.action = np.array([0.0])
        # Else keep previous action

    def _get_wind(self):
        if self.wind_std > 0.0:
            self.wind += (
                (np.random.normal(0.0, self.wind_std) - self.wind)
                * self.dt
                / (self.dt + self.wind_std)
            )

    def _get_obs(self):  #calculate what is indeed being observed
        # Compute ground truth divergence
        # State is (height, velocity, effective thrust)

        div = -2.0 * self.state[1] / max(1e-5, self.state[0])
        div_dot = (div - self.div_ph[0]) / self.dt

        # Overwrite placeholder
        self.div_ph[:] = [div, div_dot]
        # Add noise (regular and proportional)
        # Use old noisy estimate for noisy div_dot
        div += np.random.normal(0.0, self.noise_std) + abs(div) * np.random.normal(
            0.0, self.noise_p_std
        )
        div_dot = (div - self.obs[-1][0]) / self.dt

        # Append to end of deque; if == max length then first is popped
        self.obs.append([div, div_dot])

        # Return observed value or zeros if we are in the blind area
        if self.state[0] > self.h_blind:
            return np.zeros(2, dtype=np.float32)
        else:
            return np.array(self.obs[0], dtype=np.float32)

    def _clamp(self, value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def _check_done(self):
        return self._check_out_of_bounds() or self._check_out_of_time()

    def _check_out_of_bounds(self):
        return self.state[0] < self.min_h or self.state[0] > self.max_h

    def _check_out_of_time(self):
        return self.t >= self.max_t

######
    def _check_done_cost_function(self):
        return self.height_reward < self.min_h

    def _check_out_of_bounds_cost_function(self):
        return self.height_reward < self.min_h or self.height_reward > self.max_h

    def _check_out_of_time_cost_function(self): #(implicit the top one always prevails)
        return self.t >= self.max_t
######

    def _get_reward(self):
        # Use raw states because placeholder hasn't been updated yet
        # print(1.0 - np.abs(-2.0 * self.state[1] / max(1e-5, self.state[0])))
        # reward = abs( 5. - ( -2.0 * np.abs(-1.0 * self.state[1] / max(1e-5, self.state[0]))) )

    
        #this is correct, 1.0 stands for divergence, so 1 is general chosen D ORIGINALL
        # return 1.0 - np.abs(-2.0 * self.state[1] / max(1e-5, self.state[0]))  #multiply with inverse of probability of reference signal
        # delta landing speed = delta div * height/2
        # delta landing speed * delta t = height difference with optimal path.
        # return reward**2   Div/2 = landingspeed/height  ( landingspeed = (Div*height)/2 ) 

        #distance to desired trajectory:
        # height_des - current height
        
        self.height_reward =  self.height_reward - (self.ref_div*self.height_reward/2)*self.dt
        # difference = height_des - self.state[0]   # difference must be accumulated

        return np.abs(self.state[0] - self.height_reward)*self.dt
        
    def _get_max_reward(self):
        return self.max_t*self.max_h

    def _get_min_reward(self):
        return 0.0


    def _get_state_dot(self):
        # Action is delta G relative to hover G in Gs
        # So: state_dot for the first two states (height, velocity)
        # is just the last two states (velocity, action * G in m/s^2)!
        # First do nothing for some time, to allow settling of controller
        # and filling of deque 
        if self.t < self.settle:
            action = 0.0
            self.reward = 0
            self.height_reward = self.state[0]
        else:
            action = self._clamp(
                self.action, self.action_space.low[0], self.action_space.high[0]
            )

        # Thrust_dot = (new desired thrust - previous thrust) / (dt + tau_T)
        return np.array(
            [
                self.state[1],
                self.state[2],
                (action * self.G - self.state[2]) / (self.dt + self.thrust_tc),
            ],
            dtype=np.float32,
        )


    def reset(self, h0=5.0):
        # Check validity of initial height
        assert h0 >= self.min_h and h0 <= self.max_h

        # State is (height, velocity, effective thrust)
        self.state = np.array([h0, 0.0, 0.0], dtype=np.float32)


        # calculate total reward in environment
######        # new
        self.height_reward = h0
        self.reward = 0
######

        # We need a placeholder for ground truth divergence to compute div_dot
        self.div_ph = np.array([0.0, 0.0], dtype=np.float32)
        # Observations include noise, deque to allow for delay
        # Zeros are just for the initial calculation of div_dot
        self.obs = deque([[0.0, 0.0]], maxlen=self.delay + 1)

        # Other: variables that are always initialized at 0
        self.t = 0.0
        self.steps = 0
        self.wind = 0.0
        self.action = 0.0
        self.disturbance = [0.0, 0.0]
        return self._get_obs(), self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.seeds = seed
        np.random.seed(seed)
