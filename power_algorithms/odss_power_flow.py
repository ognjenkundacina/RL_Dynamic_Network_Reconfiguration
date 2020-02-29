import opendssdirect as dss
from opendssdirect.utils import Iterator
import math
import random
import pandas

class ODSSPowerFlow:
    def calculate_power_flow(self):
        dss.Solution.Solve()

    def get_losses(self):     
        return dss.Circuit.Losses()[0] / 1000 #divide to get kW value

    def get_bus_voltages(self):
        busVoltages = {}
        for busName in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(f"{busName}")
            #if (dss.Bus.kVBase() == 4.16/math.sqrt(3)):
            V_for_mean = []
            for phase in range(1, 4):
                if phase in dss.Bus.Nodes():
                    index = dss.Bus.Nodes().index(phase)
                    re, im = dss.Bus.PuVoltage()[index*2:index*2+2]
                    V = abs(complex(re, im))
                    V_for_mean.append(V)
            busVoltages.update( {busName : (sum(V_for_mean) / len(V_for_mean)) } )
        return busVoltages

    def get_network_injected_p(self):
        return dss.Circuit.TotalPower()[0]

    def get_network_injected_q(self):
        return dss.Circuit.TotalPower()[1]

    def get_lines_apparent_power(self):
        line_name_with_apparent_power = {}

        numOfLines = dss.Lines.Count()
        dss.Lines.First()
        for i in range(0, numOfLines):
            linePowers = dss.CktElement.Powers()
            firstTerminalPowers = linePowers[:len(linePowers)//2]
            p = sum(firstTerminalPowers[0::2])
            q = sum(firstTerminalPowers[1::2])
            s = math.sqrt(pow(p, 2) + pow(q, 2))
            #print(p)
            #print(q)
            #print(s)
            #print(dss.CktElement.Name())
            line_name_with_apparent_power.update( { dss.CktElement.Name() : s } )
            dss.Lines.Next()
            
        return line_name_with_apparent_power

    def get_capacitor_calculated_q(self):
        capacitor_q_injected = {}

        numOfCapacitors = dss.Capacitors.Count()
        dss.Capacitors.First()
        for i in range(0, numOfCapacitors):
            capacitorPowers = dss.CktElement.Powers()
            firstTerminalPowers = capacitorPowers[:len(capacitorPowers)//2]
            q = sum(firstTerminalPowers[1::2])
            capacitor_q_injected.update( { dss.CktElement.Name() : q } )

            dss.Capacitors.Next()

        return capacitor_q_injected

    def create_data_set(self):
        n_capacitors = dss.Capacitors.Count()
        n_consumers = dss.Loads.Count()

        columns = [i for i in range(n_consumers + n_capacitors)]
        index = [i for i in range(1000)]
        df = pandas.DataFrame(index=index, columns=columns)
        df = df.fillna(0)
        for index, row in df.iterrows():
            for i in range(n_consumers):
                df.loc[index, i] = random.random()
            for i in range(n_consumers, n_consumers + n_capacitors):
                df.loc[index, i] = random.choice([True, False])

        df.to_csv('data.csv')