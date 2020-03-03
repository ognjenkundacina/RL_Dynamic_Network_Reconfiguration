import pandas
from power_algorithms.odss_network_management import ODSSNetworkManagement
from power_algorithms.odss_power_flow import ODSSPowerFlow

def test_power_flow():
    network_manager = ODSSNetworkManagement()
    power_flow = ODSSPowerFlow()

    power_flow.calculate_power_flow()
    #print(power_flow.get_network_injected_p())

    #print('Bus voltages:')
    #print(power_flow.get_bus_voltages())
    #network_manager.testCktElement()

    #power_flow.create_data_set()