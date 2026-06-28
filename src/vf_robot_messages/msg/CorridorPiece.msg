# This message type records "instructions" for the robot to be played back. These can of various types, like "turn left" etc.
# or something like "stick to left wall". All of these also come with one float param which can mean different things, depending
# on the string.

string instruction  # one of "turn_left_90", "turn_right_90", "forward", "reverse", "end", "keep_left", "keep_right", "turn_around" (maybe more?)
float64 distance
