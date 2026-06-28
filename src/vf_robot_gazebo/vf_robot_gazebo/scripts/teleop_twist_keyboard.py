#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, termios, tty

msg = """
Control Your Robot!
---------------------------
Moving around:
   w
a  s  d
   x

w/x : forward/backward
a/d : turn left/right
space : stop

CTRL-C to quit
"""

moveBindings = {
    "w": (1.0, 0.0),
    "x": (-1.0, 0.0),
    "a": (0.0, 1.0),
    "d": (0.0, -1.0),
    " ": (0.0, 0.0),
}

speed = 0.5  # linear speed
turn = 1.0  # angular speed


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_twist_keyboard")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("Teleop node started! Use keyboard to drive.")

    def run(self):
        twist = Twist()
        print(msg)
        while rclpy.ok():
            key = getKey()
            if key in moveBindings.keys():
                linear, angular = moveBindings[key]
                twist.linear.x = linear * speed
                twist.angular.z = angular * turn
                self.pub.publish(twist)
            elif key == "\x03":  # CTRL-C
                break


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = TeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.pub.publish(twist)  # stop robot
        node.destroy_node()
        rclpy.shutdown()
