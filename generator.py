#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# import os
from collections import deque, OrderedDict
import re
import numpy as np
import pandas as pd
import scipy.linalg as linalg

DATAFILE = '/home/user/src/driver_behavior/dataset_test.csv'
# DATAFILE = '/home/user/src/git/driver_behavior/test.csv'
CSV_HEADERS = ['frame', 'steering', 'throttle', 'speed', 'brake', 'yaw',
               'vehData', 'location']


# find an angle btw ν and υ
# υ and ν are nunmpy vectors
def angle_btw(ν, υ):
    assert type(ν) == type(υ) == np.ndarray, 'v1 and v2 have to be vectors'
    v1_mag = linalg.norm(ν)
    v2_mag = linalg.norm(υ)
    cosθ = np.dot(ν, υ) / (v1_mag * v2_mag)
    θ = np.rad2deg(np.arccos(cosθ))
    return θ


# calculate rotation and translation
def rotation_trans(t, O, θ):
    assert type(t) == type(O) == np.ndarray, 'v1 and v2 have to be vectors'
    if t.shape[0] == 2:
        t = np.array([t[0], t[1], 1], dtype=np.float32)
    if O.shape[0] == 2:
        O = np.array([O[0], O[1], 1], dtype=np.float32)
    sinθ = np.sin(θ)
    cosθ = np.cos(θ)
    rot = np.array([[cosθ, -sinθ, t[0]],
                    [sinθ,  cosθ, t[1]],
                    [0,     0,    1  ]]).astype(np.float32)
    m = np.identity(3)
    m[:3, -1] = O
    outer = np.matmul(m, rot)
    return outer.astype(np.float32)


# helper function for parsing GTAV records
def parse_vehicle_log(string, yaw, my_pos):
    flag = False
    threshold = 70  # in degrees
    if string[3:6] == 'FUR':
        # string = str(string.split()[::-1])
        flag = True
        # print(flag)
    single_record = 13
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    data = rx.findall(string)
    data_length = len(data)
    my_pos = rx.findall(my_pos)
    my_pos = [float(x) for x in my_pos]
    # first calculate translation and rotation
    # after take forward vector
    v0 = rotation_trans(t=np.array([my_pos[0], my_pos[1]]),
                        O=np.array([0, 0]),
                        θ=np.deg2rad(yaw),
                        )

    # we are working in 2D
    v1 = np.sum(v0, axis=0)[:-1]
    # if there is some shit in log just return None
    if data_length % 13 != 0:
        return None
    log = deque()
    for i in range(int(data_length / 13)):
        car = Vehicle(data[:single_record], flag)
        # check if we have the same direction
        carθ = car.get_yaw()
        # get rotation and translation of this car
        # calculate difference in orientation
        # if it less then treshold, add it
        v02 = rotation_trans(t=np.array([0, 0]),
                            O=np.array([car.pos_x, car.pos_y]),
                            θ=np.deg2rad(car.get_yaw()),
                            )
        v2 = np.sum(v02, axis=0)[:-1]
        α = angle_btw(v1, v2)
        distance = linalg.norm(car.get_edge())
        disp = linalg.inv(v0).dot(v02)
        disp = np.sum(disp, axis=1)[:-1]
        if α <= threshold and float(car.speed) > 1.0 and distance <= 30:
            car.displacement = disp
            log.append(car)
        data = data[single_record:]
    # return deque with cars classes
    return log

        
def get_sequences(df, seq_len):
    data = {}
    temporal_seq = []
    sequences = []
    for i in range(df.shape[0]):
        yaw = df.yaw[i]
        my_pos = df.location[i]
        data_sample = df.vehData[i]
        dumped = parse_vehicle_log(data_sample, yaw, my_pos)
        if dumped is None:
            # print('None')
            data = OrderedDict()
            temporal_seq = []
            ptr = 0
            continue
        cars = [veh for veh in dumped]
        # check it if dump is zero
        if len(dumped) == 0:
            # re-initiate all
            data = OrderedDict()
            temporal_seq = []
            ptr = 0
            continue
        # else if storage is empty and dump isn't, add it
        elif len(temporal_seq) is 0:
            temporal_seq.append(cars)
            data[df.frame[i]] = cars
            ptr += 1
        # get more cars if conditions true
        elif len(temporal_seq) >= 1 and len(temporal_seq) < seq_len:
            if len(cars) == len(temporal_seq[ptr-1]):
                if [item.get_object_id() for item in cars] ==\
                   [item.get_object_id() for item in temporal_seq[ptr-1]]:
                    temporal_seq.append(cars)
                    data[df.frame[i]] = cars
                    ptr += 1
                else:
                    temporal_seq = []
                    data = OrderedDict()
                    ptr = 0
            else:
                temporal_seq = []
                data = OrderedDict()
                ptr = 0
        if len(temporal_seq) == seq_len:
            sequences.append(data)
            data = OrderedDict()
            temporal_seq = []
            ptr = 0
    return sequences

# calculate x_t_uu, vector from location of the node
# at time t-1 to location at time t, or just displacement?
def calculate_displacement(m):
    res = np.zeros_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1] - 1):
            res[i, j] = m[i, j + 1] - m[i, j]
    return res[::, :-1]

class Vehicle(object):
    # data input format every time change
    # TODO: notice to every time define it carefully
    def __init__(self, records=[], flag=False):
        # print(records)
        flag = True
        if flag:
            self.vehId, self.pos_x, self.pos_y, self.pos_z, self.fur_0,\
            self.fur_1, self.fur_2, self.bll_0, self.bll_1, self.bll_2,\
            self.speed, self.heading, self.classId = records
        else:
            raise NotImplemented

        self.bll = np.array([self.bll_0, self.bll_1]).astype(np.float32)
        self.fur = np.array([self.fur_0, self.fur_1]).astype(np.float32)
        self.displacement = np.array([0, 0], dtype=np.float32)

    # every data record time order of this seq is change
    # keep track seq in order to have this right one
    def __str__(self):
        str = 'PlateID: {}, Object id: {}, heading: {}, FUR: {} {} {}, speed: {}, BLL: {} {} {}, pos: {} {}'.format(
            self.vehId, self.classId, self.heading, self.fur_0, self.fur_1,
            self.fur_2, self.speed, self.bll_0, self.bll_1, self.bll_2,
            self.pos_x, self.pos_y
            )
        return str

    # edge data, represent relative position vector
    def get_edge(self):
        position = (self.bll + self.fur) / 2.0
        return np.array([position[0], position[1]], dtype=np.float32)

    # just yaw
    def get_yaw(self):
        return float(self.heading)

    # node data, represent state of vehecle
    # TODO: change input to state instead of position
    def get_node(self):
        return np.array([self.speed, self.heading], dtype=np.float32)

    # vehicle's ID
    def get_object_id(self):
        return self.vehId

    # speed of the current veh
    def get_speed(self):
        return self.speed

    def get_annotation(self):
        pass

    # prediction label
    def get_displacement(self):
        return np.array([self.displacement[0],
                         self.displacement[1]],
                         dtype=np.float32,
        )

if __name__ == '__main__':
    df = pd.read_csv(DATAFILE, names=CSV_HEADERS, sep=';')
    test = get_sequences(df, 25)
