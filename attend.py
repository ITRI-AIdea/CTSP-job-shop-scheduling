#  Copyright (c) 2020 Industrial Technology Research Institute.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from bisect import bisect_right
import pandas as pd

pd.set_option('display.max_rows', None)


class Attend(object):
    """與 attendance 表格對應"""
    def __init__(self, working):
        self.routine = working
        self.routine_today = []
        self.idle_today = []
        self.total_today = 0
        self.is_overnight = False
        self.routine_nextday = []
        self.idle_nextday = []
        self.total_nextday = 0

    @staticmethod
    def compute_idle(routine, idle):
        last = 0
        down = 0
        if not routine:
            idle = []
            return 0
        for p in routine:
            down += p[0] - last 
            idle.append(down)
            last = p[1]
        return routine[-1][1] - down

    def prepare(self):
        if self.routine_today:  # 處理過
            return self
        r = self.routine
        if not r:
            self.routine_today = self.routine
            self.total_today = Attend.compute_idle(self.routine_today, self.idle_today)
            return self

        if r[-1][1] > r[0][0]:  # 下班在上班之後
            self.routine_today = self.routine
            self.total_today = Attend.compute_idle(self.routine_today, self.idle_today)
            return self
        self.is_overnight = True
        stop_last = 0  # 最晚的時間點，下一個時間點是凌晨會變小
        to_nextday = False
        for p in self.routine:
            if to_nextday:
                self.routine_nextday.append(p)
                continue
            if p[0] < stop_last:  # 開工在隔天
                to_nextday = True
                self.routine_nextday.append(p)
                continue
            stop_last = p[0]  # 開工正常，更新最晚
            if p[1] < stop_last:  # 收工在隔天，分兩段
                to_nextday = True
                self.routine_today.append((p[0], 1440))
                self.routine_nextday.append((0, p[1]))
                continue
            self.routine_today.append(p)
            stop_last = p[1]  # 收工正常，更新最晚
        self.total_nextday = Attend.compute_idle(self.routine_nextday, self.idle_nextday)
        self.total_today = Attend.compute_idle(self.routine_today, self.idle_today)
        return self


class ManHourOneDay(object):
    def __init__(self, today, yesterday):
        self.routine = yesterday.routine_nextday.copy() if yesterday.is_overnight else []
        self.idle = yesterday.idle_nextday.copy() if yesterday.is_overnight else []

        self.routine += today.routine_today
        self.idle += [i - yesterday.total_nextday for i in today.idle_today]

        self.total = yesterday.total_nextday + today.total_today
        
        self.sections = [t for p in self.routine for t in p]
        self.stops = [p[1] for p in self.routine]

    def valid(self, target):
        sect = bisect_right(self.sections, target)
        validity = False
        if target in self.stops:  # last point of prior section
            validity = True
            return validity, sect-1
        if sect % 2 == 1:
            validity = True
            return validity, sect
        return validity, sect

    def to_mh(self, minute_of_day):
        mh = MHCoord(self, minute_of_day)
        (mh.valid, mh.sect) = self.valid(minute_of_day)
        return mh


class MHCoord(object):
    """工作時間座標相關"""
    def __init__(self, day_routine, minute_of_day):
        self.routine = day_routine
        self.target = minute_of_day
        self.valid = False
        self.sect = -1
        self.mhcoord = -1
        self.left = -1

    def to_available_prev(self):
        if self.valid:
            return self.valid
        if self.sect == 0:
            return self.valid
        self.sect -= 1
        self.target = self.routine.sections[self.sect]
        self.valid = True
        return self.valid

    def to_available_next(self):
        if self.valid:
            return self.valid
        if self.sect == len(self.routine.sections):
            return self.valid
        self.target = self.routine.sections[self.sect]
        self.sect += 1
        self.valid = True
        return self.valid

    def get_mhcoord(self):
        if not self.valid:
            return self.mhcoord
        if self.mhcoord >= 0:
            return self.mhcoord
        self.mhcoord = self.target - self.routine.idle[self.sect//2]
        return self.mhcoord
    
    def get_leftover(self):
        if not self.valid:
            return self.left
        if self.left >= 0:
            return self.left
        self.left = self.routine.total - self.get_mhcoord()
        return self.left
