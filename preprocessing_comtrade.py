from osc_tools.core.comtrade_custom import Comtrade

def remove_lines_with_disallowed_names(cfg_file_path, allowed_names):
    cff = Comtrade()
    cff.load(cfg_file_path)

    new_analog_signals = []
    new_digital_signals = []

    for i in range(len(cff.analog)):
        if cff.analog_channel_ids[i] in allowed_names:
            new_analog_signals.append(cff.analog[i])

    for i in range(len(cff.digital)):
        if cff.digital_channel_ids[i] in allowed_names:
            new_digital_signals.append(cff.digital[i])

    #cff = cff.remove_disallowed_analog_names(allowed_names)
    #cff = cff.remove_disallowed_digital_names(allowed_names)

    new_cfg_file_path = cfg_file_path[:-4] + '_filtered.cfg'
    new_dat_file_path = cfg_file_path[:-4] + '_filtered.dat'
    cff.write_to_file(new_cfg_file_path, new_dat_file_path)

# Example of using the function
allowed_names = ["U | BusBar-1 | phase: A", "U | BusBar-1 | phase: B", "U | BusBar-1 | phase: C"]
cfg_file_path = 'D:/DataSet/Новая папка/0147acceb145f2b6183e3e4c3d5a8a9a.cfg'

remove_lines_with_disallowed_names(cfg_file_path, allowed_names)