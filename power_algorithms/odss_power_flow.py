import opendssdirect as dss
from opendssdirect.utils import Iterator
import math
import random
import pandas
from config import *

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

    def get_switches_apparent_power(self):
        line_name_with_apparent_power = {}

        numOfLines = dss.Lines.Count()
        dss.Lines.First()
        for i in range(0, numOfLines):
            line_name = dss.CktElement.Name()
            if ('sw' in line_name):
                linePowers = dss.CktElement.Powers()
                firstTerminalPowers = linePowers[:len(linePowers)//2]
                p = sum(firstTerminalPowers[0::2])
                q = sum(firstTerminalPowers[1::2])
                s = math.sqrt(pow(p, 2) + pow(q, 2))
                #print(p)
                #print(q)
                #print(s)
                #print(dss.CktElement.Name())
                line_name_with_apparent_power.update( { line_name : s } )
            dss.Lines.Next() #ovo mora u svakoj iteraciji da se izvrsi
            
        return line_name_with_apparent_power

    def get_lines_apparent_power(self):
        line_name_with_apparent_power = {}

        numOfLines = dss.Lines.Count()
        dss.Lines.First()
        for i in range(0, numOfLines):
            line_name = dss.CktElement.Name()
            if ('sw' in line_name):
                dss.Lines.Next()
                continue

            linePowers = dss.CktElement.Powers()
            firstTerminalPowers = linePowers[:len(linePowers)//2]
            p = sum(firstTerminalPowers[0::2])
            q = sum(firstTerminalPowers[1::2])
            s = math.sqrt(pow(p, 2) + pow(q, 2))
            #print(p)
            #print(q)
            #print(s)
            #print(dss.CktElement.Name())
            line_name_with_apparent_power.update( { line_name : s } )
            dss.Lines.Next() #ovo mora u svakoj iteraciji da se izvrsi
            
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
        n_consumers = dss.Loads.Count()

        columns = [i for i in range(NUM_TIMESTEPS * 3)]
        index = [i for i in range(2)]
        df = pandas.DataFrame(index=index, columns=columns)
        df = df.fillna(0)
        for index, row in df.iterrows():
            df.loc[index, 0] = 0.667
            df.loc[index, 1] = 0.334
            df.loc[index, 2] = 0.221
            df.loc[index, 3] = 0.622
            df.loc[index, 4] = 0.311
            df.loc[index, 5] = 0.221
            df.loc[index, 6] = 0.544
            df.loc[index, 7] = 0.272
            df.loc[index, 8] = 0.221
            df.loc[index, 9] = 0.444
            df.loc[index, 10] = 0.222
            df.loc[index, 11] = 0.221
            df.loc[index, 12] = 0.422
            df.loc[index, 13] = 0.211
            df.loc[index, 14] = 0.221
            df.loc[index, 15] = 0.511
            df.loc[index, 16] = 0.255
            df.loc[index, 17] = 0.221
            df.loc[index, 18] = 0.644
            df.loc[index, 19] = 0.322
            df.loc[index, 20] = 1.000
            df.loc[index, 21] = 0.711
            df.loc[index, 22] = 0.355
            df.loc[index, 23] = 1.000
            df.loc[index, 24] = 0.778
            df.loc[index, 25] = 0.389
            df.loc[index, 26] = 1.000
            df.loc[index, 27] = 0.778
            df.loc[index, 28] = 0.389
            df.loc[index, 29] = 0.447
            df.loc[index, 30] = 0.767
            df.loc[index, 31] = 0.767
            df.loc[index, 32] = 1.000
            df.loc[index, 33] = 0.722
            df.loc[index, 34] = 0.722
            df.loc[index, 35] = 1.000
            df.loc[index, 36] = 0.689
            df.loc[index, 37] = 0.689
            df.loc[index, 38] = 1.000
            df.loc[index, 39] = 0.689
            df.loc[index, 40] = 0.689
            df.loc[index, 41] = 1.000
            df.loc[index, 42] = 0.667
            df.loc[index, 43] = 0.334
            df.loc[index, 44] = 1.000
            df.loc[index, 45] = 0.767
            df.loc[index, 46] = 0.911
            df.loc[index, 47] = 1.000
            df.loc[index, 48] = 0.889
            df.loc[index, 49] = 1.000
            df.loc[index, 50] = 0.221
            df.loc[index, 51] = 0.933
            df.loc[index, 52] = 0.811
            df.loc[index, 53] = 0.221
            df.loc[index, 54] = 1.000
            df.loc[index, 55] = 0.811
            df.loc[index, 56] = 0.221
            df.loc[index, 57] = 0.911
            df.loc[index, 58] = 0.705
            df.loc[index, 59] = 0.221
            df.loc[index, 60] = 0.867
            df.loc[index, 61] = 0.433
            df.loc[index, 62] = 0.221
            df.loc[index, 63] = 0.778
            df.loc[index, 64] = 0.389
            df.loc[index, 65] = 0.221
            df.loc[index, 66] = 0.711
            df.loc[index, 67] = 0.355
            df.loc[index, 68] = 0.221
            df.loc[index, 69] = 0.667
            df.loc[index, 70] = 0.334
            df.loc[index, 71] = 0.221
        df.to_csv('data.csv')