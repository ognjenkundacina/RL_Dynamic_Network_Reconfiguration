import gym
from gym import spaces
import random
import numpy as np 
#from power_algorithms.power_flow import PowerFlow
from power_algorithms.odss_power_flow import ODSSPowerFlow
#import power_algorithms.network_management as nm
import power_algorithms.odss_network_management as nm

class Environment(gym.Env):
    
    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = []

        self.network_manager = nm.ODSSNetworkManagement()
        self.power_flow = ODSSPowerFlow()
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        self.state_space_dims = len(self.power_flow.get_bus_voltages()) + len(self.power_flow.get_capacitor_calculated_q())
        self.n_actions = len(self.network_manager.get_all_capacitors())
        self.n_consumers = self.network_manager.get_load_count()
        self.i_step = 0
        self.current_losses = 0.0

        self.capacitor_names = self.network_manager.get_all_capacitor_switch_names()
        self.action_indices = [i for i in range(self.n_actions)] #todo ovo bi mogao biti dictionary, ako bismo tap changere koristili, ili diskretizovane setpointe generatora
        self.capacitor_names_by_index = dict(zip(self.action_indices, self.capacitor_names))
        
    def _update_state(self):
        self.power_flow.calculate_power_flow()

        bus_voltages_dict = self.power_flow.get_bus_voltages()
        self.state = list(bus_voltages_dict.values())
        self.state += list(self.power_flow.get_capacitor_calculated_q().values())
        #line_rated_powers_dict = self.power_flow.get_line_rated_powers()
        #self.state = list(line_rated_powers_dict.values())

        if (len(self.state) != self.state_space_dims):
            print('Environment: len(self.state) != self.state_space_dims')

        return self.state

    #action: 0..n_actions
    def step(self, action):
        self.network_manager.toogle_capacitor_status(self.available_actions[action])
        self.available_actions.pop(action)
        
        next_state = self._update_state()

        reward = self.calculate_reward(action)

        self.i_step += 1

        done = (self.i_step == self.n_actions)

        if (reward < 0.0):
            done = True  #todo staviti odradjivanje vrijednosti za done u jednu liniju

        return next_state, reward, done
    
    #only for test set
    def revert_action(self, action):
        self.network_manager.toogle_capacitor_status(self.capacitor_names_by_index[action])
        next_state = self._update_state()
        self.current_losses = self.power_flow.get_losses()
        return next_state


    def calculate_reward(self, action):
        new_losses = self.power_flow.get_losses()
        losses_decrease = self.current_losses - new_losses
        reward = losses_decrease * 1000
        self.current_losses = new_losses
        #ObjectiveFunctions

        return reward

    def reset(self, consumption_percents, capacitor_statuses):
        self.network_manager.set_load_scaling(consumption_percents)
        self.network_manager.set_capacitors_initial_status(capacitor_statuses)
        
        self.power_flow.calculate_power_flow()
        bus_voltages_dict = self.power_flow.get_bus_voltages()
        self.state = list(bus_voltages_dict.values())
        self.state += list(self.power_flow.get_capacitor_calculated_q().values())
        #line_rated_powers_dict = self.power_flow.get_line_rated_powers()
        #self.state = list(line_rated_powers_dict.values())
        self.current_losses = self.power_flow.get_losses()
        self.i_step = 0
        self.available_actions = dict(zip(self.action_indices, self.capacitor_names))

        return self.state