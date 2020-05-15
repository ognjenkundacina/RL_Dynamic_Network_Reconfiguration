import os
from environment.environment import Environment
import pandas as pd
from rl_algorithms.deep_q_learning import DeepQLearningAgent
import time
from power_algorithms.odss_network_management import ODSSNetworkManagement
from power_algorithms.odss_power_flow import ODSSPowerFlow
from power_algorithms.power_flow_tester import test_power_flow


def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/data.csv')
    df = pd.read_csv(file_path)

    return df

def split_dataset(df, split_index):
    df_train = df[df.index <= split_index]
    df_test = df[df.index > split_index]

    return df_train, df_test

def main():
    #test_power_flow()
    
    df = load_dataset()
    df_train, df_test = split_dataset(df, 0)

    #environment should'n have the entire dataset as an input parameter, but train and test methods
    environment = Environment()
    #print(environment.get_number_of_switch_manipulations([1, 5, 6], [12, 13, 14]))

    print('=====================agent=====================')
    agent = DeepQLearningAgent(environment)

    for i in range (1):
        n_episodes = 100000
        print('agent training started')
        t1 = time.time()
        agent.train(df_train, n_episodes)
        t2 = time.time()
        print ('agent training finished in', t2-t1)
        agent.test(df_test)
    
if __name__ == '__main__':
    main()
