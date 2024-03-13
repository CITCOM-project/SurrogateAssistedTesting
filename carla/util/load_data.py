import pandas as pd

def pre_process_data(file_name, percent, index, filename):
    file_reader = open(file_name, 'r')
    all_data = file_reader.read()
    data_parts = all_data.split("\n")
    total_parts = len(data_parts)
    all_data_float = []
    for data in data_parts:
        if len(data) > 0:
            data_part = data.split(" [")
            data_part = data_part[1].replace("]:", ",")
            dp = list(data_part.split(","))
            dp = [float(ind) for ind in dp]
            all_data_float.append(dp)
    sorted_data = sorted(all_data_float, key=lambda x: x[index])
    file_writer = open(filename, 'w')

    for i in range(int((total_parts * percent) / 100)):
        if i>= len(sorted_data):
            break
        to_write = str(sorted_data[i]).replace('[', '').replace(']', '') + "\n"
        file_writer.write(to_write)

def create_database(all_data,clean_data):
    pre_process_data(all_data, 100, 16, clean_data)

    dataset = pd.read_csv(clean_data, header=None)
    dataset.columns = [
        "road_type",
        "road_id",
        "scenario_length",
        "vehicle_front",
        "vehicle_adjacent",
        "vehicle_opposite",
        "vehicle_front_two_wheeled",
        "vehicle_adjacent_two_wheeled",
        "vehicle_opposite_two_wheeled",
        "time",
        "weather",
        "pedestrian_density",
        "target_speed",
        "trees",
        "buildings",
        "task",
        "follow_center",
        "avoid_vehicles",
        "avoid_pedestrians",
        "avoid_static",
        "abide_rules",
        "reach_destination",
    ]
    
    return dataset