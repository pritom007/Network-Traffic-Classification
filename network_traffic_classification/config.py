"""Shared dataset schema for the traffic classification experiments."""

DATASET_COLUMNS = [
    "#flow_id",
    "protocol",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "ndpi_proto_num",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_packets",
    "dst2src_bytes",
    "ndpi_proto",
    "class",
]

PREDICTION_COLUMNS = [
    "#flow_id",
    "protocol",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "ndpi_proto_num",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_packets",
    "dst2src_bytes",
    "ndpi_proto",
]

FEATURE_COLUMNS = [
    "protocol",
    "src_port",
    "dst_port",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_packets",
    "dst2src_bytes",
]
