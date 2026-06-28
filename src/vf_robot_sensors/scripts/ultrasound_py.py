#!/usr/bin/env python3
#
# Copyright  EUROKNOWS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Pravin Oli
# Email: pravin.oli.08@gmail.com, olipravin18@gmail.com
# Company: EUROKNOWS CO., LTD.
# Website: https://www.euroknows.com/en/home/
#
# Erasmus Mundus Joint Masters in Intelligent Field Robotics System (IFROS)
# https://ifrosmaster.org/
#
# Universitat de Girona, Spain - https://www.udg.edu/en/
# Eötvös Loránd University, Hungary - https://www.elte.hu/
#

import rclpy
from rclpy.node import Node
from vfmessages.msg import UltraSound
from sensor_msgs.msg import Range


class Ultrasound(Node):
    def __init__(self):
        super().__init__("ultrasound")  # Node name
        self.get_logger().info("Ultrasound: initializing...")

        # Publisher
        self.ultrasound_publisher = self.create_publisher(UltraSound, "/esp/range", 1)

        # Subscribers
        self.sensor_ids = {
            "front_left": 0,
            "front_right": 1,
            "right": 2,
            "rear": 3,
            "left": 4,
        }

        topics = [
            "/ultrasound/front_left",
            "/ultrasound/front_right",
            "/ultrasound/right",
            "/ultrasound/rear",
            "/ultrasound/left",
        ]

        self.subs = []
        for topic in topics:
            sub = self.create_subscription(Range, topic, self.ultrasound_callback, 1)
            self.subs.append(sub)

    def ultrasound_callback(self, msg: Range):
        # Map frame_id to sensor code
        sensor_name = msg.header.frame_id
        code = self.sensor_ids.get(sensor_name, -1)

        vfm = UltraSound()
        vfm.code = code
        vfm.range = msg.range

        self.ultrasound_publisher.publish(vfm)


def main(args=None):
    rclpy.init(args=args)
    node = Ultrasound()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
