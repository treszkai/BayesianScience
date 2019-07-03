def parse_second_column(data_str):
    return [float(line.split(", ")[1]) for line in data_str.split("\n")]
