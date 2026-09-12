OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
INDEX_COLS = ["unit_number", "time_cycles"]
ALL_COLS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS

# Sensors that are constant (or near-constant) in FD001, identified from EDA.
# Dropped in feature engineering since they carry no signal in that subset.
CONSTANT_SENSORS_FD001 = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]
