import collections
import numpy as np


class AngleBuffer:
    """
    머리 각도 값의 이동 평균을 계산하는 버퍼 클래스
    """
    def __init__(self, size=3):
        self.size = size
        self.buffer = []
        
    def add(self, angle):
        """
        새로운 각도 값을 버퍼에 추가
        """
        self.buffer.append(angle)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
            
    def get_average(self):
        """
        버퍼에 있는 각도들의 평균 반환
        """
        if not self.buffer:
            return 0
        return sum(self.buffer) / len(self.buffer)