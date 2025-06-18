import re
import numpy as np


class FrameData:
    def __init__(self, file):
        self.first_frame = 0
        self.frame_data = self._get_data_from_file(file)
        self.frame_count = len(self.frame_data)
        self.time_length = self.get_data_with_identifier("TIME",self.frame_count-1)[1]

    def get_data_with_identifier(self,identifier,frame, forward = False):
        for i in range(frame,self.frame_count) if forward else range(frame,0,-1):
            if self.frame_data[i] is not None and identifier in self.frame_data[i]:
                return i, self.frame_data[i][identifier]
        return -1, None

    def get_all_frames_with_identifier(self, identifier):
        res = []
        for i in range(0,self.frame_count):
            if self.frame_data[i] is not None and identifier in self.frame_data[i]:
                res.append(self.frame_data[i][identifier])

        return res

    def get_closest_frame_to_timestamp(self,timestamp):
        for i in range(self.frame_count):
            if self.frame_data[i] is None: continue
            if self.frame_data[i]["TIME"] > timestamp:
                return max(0, i - 1 - self.first_frame)
        return -1

    def _read_lines_from_file(self, input_file):
        with open(input_file, "r") as f:
            content = f.read().split("\n")
        return content


    def _extract_data(self, data):
        type_info, value = data.split(":")

        type, name = type_info.split("_")

        if type == "POS":
            res = np.array([float(e) for e in value[1:-1].split(",")])

            if len(res) == 3:
                res[2] *= -1

            return name, res

        elif type == "S":
            return name, value

        elif type == "LIST":
            return name, [e for e in value.split(",")]

        elif type == "NUM":
            return name, int(value)

        else:
            print(f"Unknown type {type} with name {name} and value {value}")

            return None, None

    def _get_data_from_file(self, file):
        is_event = False
        lines = self._read_lines_from_file(file)

        match = re.match(r"\[.*\]", lines[-2])

        elements = match.group().split("|")

        frame_count = int(elements[2].replace("]", ""))

        frame_data = [None] * (frame_count + 1)
        for line in lines:
            match = re.match(r"\[.*\]", line)

            if not match:
                print(f"Invalid line: {line}")
                continue

            header_elements = match.group().split("|")

            if len(elements) < 3:
                print(f"Invalid header: {line}")
                continue

            time = int(header_elements[0][1:])

            identifier = header_elements[1]

            frame_num = int(header_elements[2][:-1])

            if frame_data[frame_num] is None:
                frame_data[frame_num] = {}

            input_data = line[match.end():].split(";")

            frame_data[frame_num]["FRAME_NUM"] = frame_num
            frame_data[frame_num]["TIME"] = time

            if identifier not in frame_data[frame_num]:
                frame_data[frame_num][identifier] = {}

            if is_event:
                event_num += 1
            else:
                event_num = 0

            is_event = False

            for data in input_data:
                name, val = self._extract_data(data)
                if name == "EVENT":
                    is_event = True
                    if type(frame_data[frame_num][identifier]) != type([]):
                        frame_data[frame_num][identifier] = []

                if is_event:
                    for i in range(event_num-len(frame_data[frame_num][identifier]) + 1):
                        frame_data[frame_num][identifier].append({})
                    frame_data[frame_num][identifier][event_num][name] = val
                else:
                    frame_data[frame_num][identifier][name] = val

        for frame in frame_data[1:]:
            if frame is not None:
                print("meep")
                print(frame)
                self.first_frame = frame["FRAME_NUM"]
                print(self.first_frame)
                break
        return frame_data