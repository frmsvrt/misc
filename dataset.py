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
from deepgtav.client import Client
from deepgtav.messages import Start, Stop, Scenario, frame2numpy, Dataset
import argparse
import cv2
import os
DATASET_PATH = '/home/user/data/gtav3'
from skimage.transform import resize
import matplotlib.image as mimg

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='10.60.124.42',
                     help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000,
                     help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default='dataset.pz',
                     help='Place to store the dataset')
    parser.add_argument('-s', '--save_data', default=False,
                     help='Record data?')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port

    client = Client(ip=args.host, port=args.port)
    # Dataset options
    dataset = Dataset(
            rate=10, 
            frame=[640, 320],
            throttle=True, 
            brake=True, 
            steering=True,
            location=True, 
            speed=True, 
            yawRate=True,
            vehicles=True,
            direction=[-2573.13916015625, 2000, 13.241103172302246]
            )

    # Automatic driving scenario
    scenario = Scenario(
            weather='EXTRASUNNY',
            vehicle='blista',
            time=[12,0],
            drivingMode=[1,123],
            location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]
            )
    client.sendMessage(Start(scenario=scenario,dataset=dataset)) # Start request
    print('Loading stuff.. Finish.')
    count = 0

    old_location = [0, 0, 0]
    old_speed = 0.0
    i = 96876

    while True: # Main loop
        try:
            # Message recieved as a Python dictionary
            message = client.recvMessage()
            # frame = np.resize(np.fromstring(message['frame'],
            # dtype=np.float64), (320, 160, 3))
            frame = frame2numpy(message['frame'], (640,320))
            frame = frame[10:300,:,:]
            frame = (resize(frame, (160,320)) * 255.0).astype('uint8')
            frame_name = "frame_%d.jpg" % count
            # cv2.imwrite(os.path.join(DATASET_PATH, frame_name), frame)
            if args.save_data:
                mimg.imsave(os.path.join(DATASET_PATH, frame_name), frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            throttle = message['throttle']
            speed = message['speed']
            # direction = message['direction']
            yawRate = message['yawRate']
            brake = message['brake']
            steering = message['steering']
            location = message['location']
            vehicles = message['vehicles']
            direction = message['direction']
            # print(vehicles)
            # print(location)

            if args.save_data:
                cv2.imwrite(os.path.join(DATASET_PATH, frame_name), frame)
                with open('./dataset_3.csv', 'a') as f:
                    f.write("%s/frame_%d.jpg;" % (DATASET_PATH, count))
                    f.write(str(steering) + ';')
                    f.write(str(throttle) + ';')
                    f.write(str(speed) + ';')
                    f.write(str(brake) + ';')
                    f.write(str(yawRate) + ';')
                    f.write(str(vehicles) + ';')
                    f.write(str(location) + '\n')
                    # f.write(str(*location) + '\n')
                    # f.write(str(direction) + '\n')

            # throttle steering brake
            # client.sendMessage(Commands(.0, -1.0, 1.0))
            print('\r', 'Steering: %.3f ' % steering, count, end='' )
            count += 1
        except KeyboardInterrupt:
            j = input('Paused. Press p to continue and q to exit... ')
            if j == 'p':
                continue
            elif j == 'q':
                cv2.destroyAllWindows()

    # DeepGTAV stop message
    client.sendMessage(Stop())
    client.close()
