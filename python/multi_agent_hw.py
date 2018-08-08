#!/usr/bin/env python

import rosbag
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import math


def get_edges(plot):
    inbag = rosbag.Bag('/home/brendon/Documents/backend_optimizer/python/bags/fletcher.bag')
    # THIS PARSES THE double_loop_fletcher.bag (something like that) in the directory in the j drive group folder
    # groups/magicc-data/old_server/groups/relative_nav/rosbag/FLETCHER

    msgs = []
    for topic, msg, t in tqdm(inbag.read_messages(), total=inbag.get_message_count()):
        if topic == "/edge" or topic == "/loop_closure":
            msgs.append(msg)

    return 1


if __name__ == "__main__":
    plot = True
    i = get_edges(plot)