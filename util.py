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

from typing import Tuple, Any
from bisect import bisect

import json
from datetime import datetime, timedelta
from dateutil import parser

import pandas as pd
from attend import (Attend, ManHourOneDay)

cols_bom = [
    "productCode",
    "sequence",
    "operationCode",
    "resourceCode",
    "prepareTime",
    "operationTime",
]

cols_submit = [
    "orderCode",
    "productCode",
    "sequence",
    "operationCode",
    "resourceCode",
    "resourceUsage",
    "start",
    "end",
]

tab_unit = {"H": 60, "M": 1}


class JobShop(object):
    def __init__(self, jobs_data):
        # convert to DataFrame
        self.df_resource = pd.DataFrame.from_dict(jobs_data["resource"])
        # ["resourceCode", "weekday", "attendanceCode", "quantity", "usageMin", "usageMax"]

        self.df_attend = pd.DataFrame.from_dict(jobs_data["attendance"])
        # ["attendanceCode", "start", "end"]

        self.df_order = pd.DataFrame.from_dict(jobs_data["order"])
        # ["orderCode", "productCode", "notBefore", "notAfter", "quantity"]

        self.df_op = pd.DataFrame.from_dict(jobs_data["BOM"])

        self.df_op = self.df_op[cols_bom]

        self.df_op_rc = {}
        for rc, _df in self.df_op.groupby(['resourceCode']):
            idx = pd.MultiIndex.from_frame(_df[['productCode', 'sequence']])
            df = _df.set_index(idx)
            self.df_op_rc[rc] = df

        # table to lookup op count for each product
        self.tab_prod_op_count = self.df_op["productCode"].value_counts().to_dict()
        # print(tab_prod_op_count)
        # {'PD004': 32, 'PD005': 31, 'PD002': 20, 'PD001': 19, 'PD003': 14, 'PD000': 8}

        self.tab_prod_op_count_nodupe = count_product_ops(self.df_op)
        # print(tab_prod_op_count_nodupe)
        # {'PD000': 8, 'PD001': 15, 'PD002': 15, 'PD003': 13, 'PD004': 24, 'PD005': 24}

        self.tab_order_qty = pd.Series(self.df_order.quantity.values, index=self.df_order.orderCode).to_dict()
        self.tab_rc_qty = {
            rc: _df.quantity.max() for rc, _df in self.df_resource.groupby(["resourceCode"])
        }  # 資源總人數

        self.tab_time_constraint = self.df_order[["notBefore", "notAfter"]].set_index(self.df_order.orderCode).to_dict(
            'index')
        # {'C183-1': {'notBefore': '2020-08-01T00:00', 'notAfter': '2020-11-23T00:00'}, 
        #  'C183-2': {'notBefore': '2020-08-01T00:00', 'notAfter': '2020-11-23T00:00'},
        # }

        self.df_up = None
        self.opcnt_total = -1

    def set_submit(self, submit_data):
        self.df_up = submit_data
        self.opcnt_total = count_ops_total(self)
        # print(f"ops needed {self.opcnt_total}")


def to_minute_of_day(s):
    """ from hh:mm string to minute of day, no check """

    _ = s.partition(':')
    return int(_[0]) * 60 + int(_[2])


def import_data(jobs: str) -> Tuple[bool, Any]:
    result = None
    try:
        with open("jobs.json", "r") as fin:
            result = json.load(fin)
        return True, result
    except Exception:
        msg = "load environment failed."
    return False, msg


def calc_rest(df):
    rest_accu = 0
    last_end = 0
    _bag = []
    for _, r in df.iterrows():
        if r['weekday'] == 1:
            _bag.append(90)  # compensate for adjust
            continue
        rest = r['mod_start'] - last_end
        last_end = r['mod_end']
        rest_accu += rest
        _bag.append(rest_accu)
    df['rest'] = _bag


worktime_daily = {}
worktime_daily_debug = {}
attends = {}


def init_daily_template(attendances):
    attendances['mod_start'] = attendances['start'].apply(to_minute_of_day)
    attendances['mod_end'] = attendances['end'].apply(to_minute_of_day)
    global worktime_daily, worktime_daily_debug, attends
    # >>> new code
    attends = {}
    attends['dummy'] = Attend([]).prepare()
    for n, g in attendances.groupby(['attendanceCode']):
        zones = list(g[['mod_start', 'mod_end']].itertuples(index=False, name=None))
        attends[n] = Attend(zones).prepare()

    worktime_daily = {}
    worktime_daily_debug = {}
    # attendanceCode,  start,    end
    for n, g in attendances.groupby(['attendanceCode']):
        zones = list(g[['mod_start', 'mod_end']].itertuples(index=False, name=None))
        attends[n] = Attend(zones).prepare()
        d = g.copy()
        d.columns = ['weekday', 'start', 'end', 'mod_start', 'mod_end']
        d['weekday'] = 0
        d.reset_index(inplace=True, drop=True)
        if n == 'TM00':
            r = d.iloc[-1]
            r2 = [[0, r[1], '24:00', r[3], 1440], [1, '00:00', r[2], 0, r[4]]]
            d = d.drop(labels=[2]).append(pd.DataFrame(r2, columns=d.columns)).reset_index(drop=True)

            # print(d.add([1,'','']))  # adjust to week day

        calc_rest(d)
        # print(d)
        worktime_daily_debug[n] = d
        worktime_daily[n] = d.iloc[:, [0, 3, 4, 5]]


rc_weekly = {}
rc_week_total = {}
onedays = {}
worktime_weekly = {}


def init_worktime(cf):
    resources = cf.df_resource
    attendances = cf.df_attend
    init_daily_template(attendances)
    # print(worktime_daily.keys())

    global worktime_weekly, rc_weekly, onedays, rc_week_total

    rc_weekly = {}
    # resourceCode, weekday, attendanceCode

    for n, g in resources.groupby(['resourceCode']):
        # print(n)  # groupby resourceCode
        modes_weekly = {i: 'dummy' for i in range(8)}
        for _, r in g.iterrows():
            mode = r.attendanceCode

            # expand workday to list contains workday
            workday = r.weekday  # '1-5', '1-6', '6'
            workday = workday.partition('-')
            if workday[2]:
                workday = list(range(int(workday[0]), int(workday[2]) + 1))
            else:
                workday = [int(workday[0])]
            for i in workday:
                modes_weekly[i] = mode
        week_table = rc_weekly.setdefault(n, {})
        total = 0
        for wd in range(1, 8):
            k = (modes_weekly[wd], modes_weekly[wd - 1])
            if k not in onedays:
                onedays[k] = ManHourOneDay(attends.get(k[0]), attends.get(k[1]))
            day_routine = onedays[k]
            total += day_routine.total
            week_table[wd] = day_routine
        rc_week_total[n] = total

    worktime_weekly = {}

    # resourceCode, weekday, attendanceCode
    for n, g in resources.groupby(['resourceCode']):
        # print(n)  # groupby resourceCode
        _bag_whole_code = []
        compensate = -90 if 'TM00' in g['attendanceCode'].unique() else 0
        # print(compensate)
        for _, r in g.iterrows():
            mode = r.attendanceCode

            # expand workday to list contains workday
            workday = r.weekday  # '1-5', '1-6', '6'
            workday = workday.partition('-')
            if workday[2]:
                workday = list(range(int(workday[0]), int(workday[2]) + 1))
            else:
                workday = [int(workday[0])]
            _bag_a_record = []
            template = worktime_daily[mode]
            for i in workday:
                if i == 1:
                    d = template.copy().add([i, 0, 0, 0])
                    if mode == 'TM00':
                        d.iat[-1, 3] = 0
                    _bag_a_record.append(d)
                    continue
                _bag_a_record.append(template.copy().add([i, 0, 0, compensate]))
            df_worktime = pd.concat(_bag_a_record)
            _bag_whole_code.append(df_worktime)

        dfnew = pd.concat(_bag_whole_code).reset_index(drop=True)
        for w, partial in dfnew.groupby(['weekday']):
            _df = worktime_weekly.setdefault(n, {})
            _df[w] = partial.reset_index(drop=True)
        # worktime_weekly[n] = pd.concat(_bag_whole_code).reset_index(drop=True)


day_worktime = {}


def calc_day_worktime():
    global day_worktime
    for rc, tab_rc in worktime_weekly.items():
        wt = day_worktime.setdefault(rc, {})
        for wd in range(1, 8):
            tt = tab_rc.get(wd)
            if not isinstance(tt, pd.DataFrame):
                wt[wd] = 0
                continue
            wt[wd] = tt.iat[-1, 2] - tt.iat[-1, 3]


def get_worktime(rc: str):
    return day_worktime.get(rc)


def calc_midday_worktime(worktime, first_day: int, count_mid_day: int) -> int:
    week = list(range(1, 8))
    mh_day = [worktime[wd] for wd in week]
    days_first_week = 7 - first_day  # first_day 當週，之後還有幾天
    cnt = count_mid_day
    if cnt < days_first_week:  # 不到一週
        return sum(mh_day[first_day:first_day + cnt])
    parts = [0] * 3
    parts[0] = sum(mh_day[first_day:])  # 第一週
    cnt -= days_first_week
    if cnt >= 7:  # 超過一週
        weeks = cnt // 7
        parts[1] = sum(mh_day) * weeks
    cnt = cnt % 7
    parts[2] = sum(mh_day[:cnt])
    return sum(parts)


def valid(resource: str, weekday: int, timestamp: datetime) -> Tuple[bool, int, int]:
    """return Tuple[valid:bool, minute_of_day:int, rest_time:int]"""

    timetable = worktime_weekly.get(resource).get(weekday)

    minod = timestamp.hour * 60 + timestamp.minute
    if not isinstance(timetable, pd.DataFrame):
        print(f"timetable not dataframe")
        return False, minod, -1

    # print(minod)
    section = [p for _, sub in timetable.iloc[:, [1, 2]].iterrows() for p in sub]

    periods = bisect(section, minod)
    # print((minod, periods, periods//2, self._section))

    # after last valid section
    if periods == len(section) and minod != section[-1]:
        return False, minod, -1

    # before first valid section
    if periods == 0:
        return False, minod, -1
    # print(periods//2)
    if periods % 2 == 1:
        return True, minod, timetable.iloc[(periods // 2), 3]
    # check section right open point      
    if section[periods - 1] == minod:
        return True, minod, timetable.iloc[(periods // 2) - 1, 3]
    return False, minod, -1


def calc_diff_old(resource: str, t1: datetime, t2: datetime) -> int:
    """return -1 for error, assume t2>t1 ordered"""

    weektable = worktime_weekly.get(resource)
    if not weektable:
        print("no weektable")
        return -1
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    timetable = [weektable.get(w) for w in wday]
    minofday = []
    restinday = []

    for wd, ts in zip(wday, tstamp):
        ok, mod, rest = valid(resource, wd, ts)
        if not ok:
            print(f"not ok {wd}, {ts}")
            return -1
        minofday.append(mod)
        restinday.append(rest)

    # print(minofday)
    # print(restinday)
    if t1.date() == t2.date():
        return (minofday[1] - restinday[1]) - (minofday[0] - restinday[0])

    partition = [0] * 3
    # calc firstday left over
    # print(timetable[0].iloc[-1])
    tt = timetable[0]
    partition[0] = (tt.iat[-1, 2] - tt.iat[-1, 3]) - (minofday[0] - restinday[0])
    # calc lastday works
    partition[2] = minofday[1] - restinday[1]
    # calc midday
    cnt_mid_day = (t2.date() - t1.date()).days - 1
    # print(cnt_mid_day)
    wt = get_worktime(resource)
    partition[1] = calc_midday_worktime(wt, wday[0], cnt_mid_day)
    # print(partition)
    return sum(partition)


def calc_diff_fullday(resource: str, t1: datetime, t2: datetime) -> int:
    """return -1 for error, assume t2>t1 ordered"""
    rc_routines = rc_weekly.get(resource)

    if not rc_routines:
        print("no weektable")
        return -1
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    # print(wday)
    timetable = [rc_routines.get(w) for w in wday]
    mod = [t.hour * 60 + t.minute for t in tstamp]
    mhcoord = [timetable[0].to_mh(mod[0]), timetable[1].to_mh(mod[1])]
    restinday = [mh.get_leftover() for mh in mhcoord]
    minofday = [mh.get_mhcoord() for mh in mhcoord]

    if any(mh == -1 for mh in minofday):
        return -1

    days = (t2.date() - t1.date()).days

    if minofday[0] == 0:  # 開工前
        days += 1
    if restinday[1] > 0:  # 收工前，不滿一天
        days -= 1
    return days


def calc_diff(resource: str, t1: datetime, t2: datetime) -> int:
    """return -1 for error, assume t2>t1 ordered"""
    rc_routines = rc_weekly.get(resource)

    if not rc_routines:
        print("no weektable")
        return -1
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    # print(wday)
    timetable = [rc_routines.get(w) for w in wday]
    mod = [t.hour * 60 + t.minute for t in tstamp]
    mhcoord = [timetable[0].to_mh(mod[0]), timetable[1].to_mh(mod[1])]
    restinday = [mh.get_leftover() for mh in mhcoord]
    minofday = [mh.get_mhcoord() for mh in mhcoord]

    if any(mh == -1 for mh in minofday):
        return -1

    if t1.date() == t2.date():
        return (minofday[1] - minofday[0])

    partition = [0] * 3
    # calc firstday left over
    # print(timetable[0].iloc[-1])
    partition[0] = restinday[0]
    # calc lastday works
    partition[2] = minofday[1]
    # calc midday
    cnt_mid_day = (t2.date() - t1.date()).days - 1
    # print(cnt_mid_day)
    wt = get_worktime(resource)
    partition[1] = calc_midday_worktime(wt, wday[0], cnt_mid_day)
    # print(partition)
    return sum(partition)


def calc_diff_available(resource: str, t1: datetime, t2: datetime) -> int:
    """return -1 for error, assume t2>t1 ordered"""
    rc_routines = rc_weekly.get(resource)

    if not rc_routines:
        print("no weektable")
        return -1
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    # print(wday)
    timetable = [rc_routines.get(w) for w in wday]
    mod = [t.hour * 60 + t.minute for t in tstamp]
    mhcoord = [timetable[0].to_mh(mod[0]), timetable[1].to_mh(mod[1])]
    while not mhcoord[0].to_available_next():
        t1 = datetime.combine(t1.date(), datetime.min.time()) + timedelta(days=1)
        tbl = rc_routines.get(t1.isoweekday())
        mhcoord[0] = tbl.to_mh(0)
    if t2 < t1:
        print(f"rc={resource}, start={str(t1)}, end={str(t2)}")
        print("wrong order")
        return -1
    while not mhcoord[1].to_available_prev():
        # 今天前面沒時間
        if t2.time() != datetime.min.time():
            t2 = datetime.combine(t2.date(), datetime.min.time())  # 今天開始
            t2 += timedelta(minutes=-1)  # 昨天 23：59
        else:
            t2 += timedelta(days=-1)  # 往前一天 
        tbl = rc_routines.get(t2.isoweekday())
        mhcoord[1] = tbl.to_mh(t2.hour * 60 + t2.minute)
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    mod = [t.hour * 60 + t.minute for t in tstamp]

    restinday = [mh.get_leftover() for mh in mhcoord]
    minofday = [mh.get_mhcoord() for mh in mhcoord]

    if any(mh == -1 for mh in minofday):
        return -1

    if t1.date() == t2.date():
        return (minofday[1] - minofday[0])

    partition = [0] * 3
    # calc firstday left over
    # print(timetable[0].iloc[-1])
    partition[0] = restinday[0]
    # calc lastday works
    partition[2] = minofday[1]
    # calc midday
    cnt_mid_day = (t2.date() - t1.date()).days - 1
    # print(cnt_mid_day)
    wt = get_worktime(resource)
    partition[1] = calc_midday_worktime(wt, wday[0], cnt_mid_day)
    # print(partition)
    return sum(partition)


def debug_calc_diff(resource: str, t1: datetime, t2: datetime) -> int:
    """return -1 for error, assume t2>t1 ordered"""
    weektable = worktime_weekly.get(resource)
    if not weektable:
        print("no weektable")
        return -1
    tstamp = [t1, t2]
    wday = [t.isoweekday() for t in tstamp]
    timetable = [weektable.get(w) for w in wday]
    minofday = []
    restinday = []

    for wd, ts in zip(wday, tstamp):
        ok, mod, rest = valid(resource, wd, ts)
        if not ok:
            print(f"not ok {wd}, {ts}")
            return -1
        minofday.append(mod)
        restinday.append(rest)

    # print(minofday)
    # print(restinday)
    if t1.date() == t2.date():
        return (minofday[1] - restinday[1]) - (minofday[0] - restinday[0])

    partition = [0] * 3
    # calc firstday left over
    # print(timetable[0].iloc[-1])
    tt = timetable[0]
    partition[0] = (tt.iat[-1, 2] - tt.iat[-1, 3]) - (minofday[0] - restinday[0])
    # calc lastday works
    partition[2] = minofday[1] - restinday[1]
    # calc midday
    cnt_mid_day = (t2.date() - t1.date()).days - 1
    # print(cnt_mid_day)
    wt = get_worktime(resource)
    partition[1] = calc_midday_worktime(wt, wday[0], cnt_mid_day)
    # print(partition)
    return sum(partition)


def check_monotonic(lst) -> bool:
    v1 = lst[0]
    for v2 in lst[1:]:
        if v2 < v1:
            return False
        v1 = v2
    return True


def count_product_ops(df_op):
    """計算每個品種的工序數量，回傳 dict[productCode:str, op_count:int]"""
    _df = df_op.sequence
    _df.index = df_op.productCode
    _df = _df.groupby(_df.index).nunique()
    # {'PD000': 8, 'PD001': 15, 'PD002': 15, 'PD003': 13, 'PD004': 24, 'PD005': 24}
    return _df.to_dict()


def count_ops_total(cf) -> int:
    df_order = cf.df_order
    df_op_tbl = df_order[["orderCode", "productCode"]]
    srs_opcnt = df_op_tbl.productCode.apply(cf.tab_prod_op_count_nodupe.get)
    return df_order.quantity.dot(srs_opcnt)


def check_product_op_count(cf, early_stop=False) -> Tuple[bool, str]:
    op_cnt_up = cf.df_up["orderCode"].value_counts().to_dict()
    # print(op_cnt_up)
    errors = []

    for order, df_up_order in cf.df_up.groupby(["orderCode"]):
        prod = df_up_order.productCode.iat[0]
        # print(f"{order}:{prod}")
        order_qty = cf.tab_order_qty.get(order)
        prod_op_count = cf.tab_prod_op_count_nodupe.get(prod)
        prod_up_count = op_cnt_up.get(order)
        # compare total ops in order
        needed = order_qty * prod_op_count
        found = prod_up_count
        if found != needed:
            # find dismatch
            msg = f"find {found}/{needed} ops, invalid."
            if early_stop:
                return False, msg
            errors.append(f"{order} {msg}")
        # compare qty
        op_count = df_up_order["sequence"].value_counts().values
        if not (op_count == order_qty).all():
            msg = f"contains wrong number of records."
            if early_stop:
                return False, msg
            errors.append(f"{order} {msg}")

    if not errors:
        return True, "record conut OK"
    return False, errors


def dump_routine(r) -> str:
    msg = []
    for k, v in r.items():
        msg.append(f"{k}:{v.__dict__}")
    return str(msg)


def check_op_time(cf, early_stop=False):
    errors = []

    for _, r in cf.df_up.iterrows():
        # print(r.columns)
        # ["orderCode", "productCode", "sequence", "operationCode",
        #  "resourceCode", "resourceUsage", "start", "end"]

        prod = r.productCode
        seq = r.sequence
        rc = r.resourceCode
        usage = r.resourceUsage
        start_u = r.start
        end_u = r.end
        (ok, op_sol) = get_operationData(cf, prod, seq, rc)  # from sol
        if not ok:
            # print(time_prep)
            errors.append(str(op_sol))
            msg = f"bad record. {str(r.to_dict())}"
            if early_stop:
                return False, msg
            errors.append(msg)
            continue
        dt_start_u = parser.isoparse(start_u)
        dt_end_u = parser.isoparse(end_u)
        time_op_raw = op_sol.get("operationTime")
        time_prepare = op_sol.get("prepareTime")
        if time_op_raw.endswith("D"):
            # 處理 "D" 資料
            if time_prepare > 0:
                # TODO
                raise Exception("D days")

            _days = int(time_op_raw[:-1])  # from sol

            days_diff = calc_diff_fullday(rc, dt_start_u, dt_end_u)
            if days_diff < _days:
                errors.append(time_op_raw)
                msg = f"not enough time {str(r.to_dict())}"
                if early_stop:
                    return False, msg
                errors.append(msg)
                # errors.append(f"start {dt_start_u.isoweekday()} {datetime2minod(dt_start_u)}")
                # errors.append(f"end {dt_end_u.isoweekday()} {datetime2minod(dt_end_u)}")
                # routine = rc_weekly.get(rc)
                # errors.append(dump_routine(routine))
            continue
        # 其他正常資料 （"H", "M"）
        # print(time_op_raw)
        totalTime = int(time_op_raw[:-1]) * tab_unit.get(time_op_raw[-1]) + time_prepare
        diffTime = calc_diff(rc, dt_start_u, dt_end_u)

        if diffTime < 0:
            msg = f"invalid start/end time: {str(r.to_dict())}"
            if early_stop:
                return False, msg
            errors.append(msg)
            # errors.append(f"start {dt_start_u.isoweekday()} {datetime2minod(dt_start_u)}")
            # errors.append(f"end {dt_end_u.isoweekday()} {datetime2minod(dt_end_u)}")
            # routine = rc_weekly.get(rc)
            # errors.append(dump_routine(routine))
            # errors.append(f"diffTime {diffTime}, totalTime {totalTime}")
            continue

        if diffTime * usage < totalTime:
            msg = f"allocate not enough time {str(r.to_dict())}"
            if early_stop:
                return False, msg
            errors.append(msg)
            # errors.append(f"start {dt_start_u.isoweekday()} {datetime2minod(dt_start_u)}")
            # errors.append(f"end {dt_end_u.isoweekday()} {datetime2minod(dt_end_u)}")
            # routine = rc_weekly.get(rc)
            # errors.append(dump_routine(routine))
            # errors.append(f"diffTime {diffTime}, totalTime {totalTime}")
            continue

    if not errors:
        return True, "OK"
    return False, errors


def get_operationData(cf, prod: str, seq: int, rc: str) -> dict:
    # r = df_op[df_op["productCode"] == prod]
    # r = r[r["sequence"] == seq]
    # r = r[r["resourceCode"] == rc]
    df = cf.df_op_rc.get(rc)
    if (prod, seq) not in df.index:
        return False, {}
    r = df.loc[(prod, seq)]
    # [{'productCode': 'PD001', 'sequence': 20, 'operationCode': 'PC024',
    #   'resourceCode': 'RC010', 'prepareTime': 0, 'TimeRaw': '90M'}]
    result = r.to_dict()
    return True, result


def check_overlap_inside_product(cf, early_stop=False):
    errors = []
    for order, df_up_order in cf.df_up.groupby(["orderCode"]):

        order_qty = cf.tab_order_qty.get(order)
        data = df_up_order[["orderCode", "sequence", "resourceCode", "start", "end"]]
        seq_first = data.sequence.min()
        seq_cur = 0
        seq_last = 0
        fin_seq = data.sort_values(["sequence", "end"]).reset_index(drop=True)
        fin_seq['fin'] = fin_seq.index % order_qty

        data = fin_seq.sort_values(["sequence", "start"]).reset_index(drop=True)
        data['startseq'] = data.index % order_qty
        idx = pd.MultiIndex.from_frame(fin_seq[['sequence', 'fin']])
        fin_seq = fin_seq.set_index(idx)

        # cnt = 30
        for idx, r in data.iterrows():
            dt_start = parser.isoparse(r.start)
            if r.sequence == seq_first:
                seq_cur = r.sequence
                seq_last = r.sequence
                continue
            if r.sequence != seq_cur:
                seq_last = seq_cur
                seq_cur = r.sequence

            rlast = fin_seq.loc[(seq_last, r.startseq)]
            dt_end = parser.isoparse(rlast.end)
            if dt_end > dt_start:  # 前工序結束在此工序開始之後
                print(df_up_order.loc[(df_up_order.sequence.isin([seq_last, seq_cur]))])
                msg = f"worktime overlapped {str(rlast.to_dict())}:{str(rlast.to_dict())}"
                if early_stop:
                    return False, msg
                errors.append('worktime overlapped')
                errors.append(f"last:{str(rlast.to_dict())}")
                errors.append(f"curr:{str(r.to_dict())}")

    if errors:
        return False, errors
    return True, ""


def check_submit(cf):
    df_up = cf.df_up
    if df_up.shape[1] != len(cols_submit):
        msg = "submit column dismatch."
        # print(msg)
        raise Exception(msg)

    if df_up.shape[0] != cf.opcnt_total:
        msg = f"submit find {df_up.shape[0]}/{cf.opcnt_total} operations."
        # print(msg)
        raise Exception(msg)
    # print(df_up.columns.to_list())

    if set(df_up.columns.to_list()) != set(cols_submit):
        msg = "submit find invalid columns."
        raise Exception(msg)


def check_rc_block(bag: list, quantity: int, early_stop=False) -> Tuple[bool, list]:
    """檢查平行資源使用量，是否超標"""
    if quantity == 0:  # infinity capacity
        return True, ""

    if not bag:
        return True, ""

    tsq = {}  # time sequence
    for r in bag:
        usage = r.resourceUsage
        v = tsq.setdefault(r.start, 0)
        tsq[r.start] = v + usage
        v = tsq.setdefault(r.end, 0)
        tsq[r.end] = v - usage

    cnt = 0
    err = []
    for _, update in sorted(tsq.items()):
        cnt += update
        if cnt > quantity:
            msg = f"run out of resource {quantity}"
            if early_stop:
                return False, f"{msg}:{str(bag[0].to_dict())}"
            err.append(msg)
            for r in bag:
                err.append(r.to_dict())
            return False, err

    return True, []


def check_rc_usage(cf, early_stop=False):
    err = []
    for rc, df_up_rc in cf.df_up.groupby(["resourceCode"]):
        rc_data = cf.df_resource[cf.df_resource.resourceCode == rc]
        rc_qty = rc_data.quantity.values[0]
        rc_min = rc_data.usageMin.values[0]
        rc_max = rc_data.usageMax.values[0]
        data = df_up_rc.sort_values(["start"]).reset_index(drop=True)
        r_last = None
        bag_connected = []
        for _, r in data.iterrows():
            if r.resourceUsage < rc_min:
                msg = f"usage less than {rc_min}:{str(r.to_dict())}"
                if early_stop:
                    return False, msg
                err.append(msg)
            elif r.resourceUsage > rc_max:
                msg = f"usage larger than {rc_max}:{str(r.to_dict())}"
                if early_stop:
                    return False, msg
                err.append(msg)
            if r_last is None:
                r_last = r
                continue
            if r.start > r_last.end:  # 非平行
                ok, msg = check_rc_block(bag_connected, rc_qty, early_stop)
                if not ok:
                    if early_stop:
                        return False, msg
                    err.append(msg)
                bag_connected = []
                r_last = r
                continue
            # 平行
            bag_connected.append(r_last)
            r_last = r

    if not err:
        return True, []
    return False, err


def check_orders(cf, early_stop=False) -> Tuple[bool, str]:
    msg = f"df_up {cf.df_up.shape}"
    # print(msg)
    order_up = cf.df_up["orderCode"].unique()
    order_sol = cf.df_order["orderCode"].unique()
    msg = f"find {len(order_up)} orders."
    # print(msg)
    if len(order_sol) != len(order_up):
        order_up = set(order_up)
        order_sol = set(order_sol)
        msg = f"order dismatch. order not found: {order_sol - order_up}, order not expected: {order_up - order_sol}"
        return False, msg
    return True, ""


def prepare(cf, submit):
    init_worktime(cf)
    calc_day_worktime()
    cf.set_submit(submit)
    try:
        check_submit(cf)
    except Exception as e:
        return False, str(e.args)
    return True, "OK"


def check(cf):
    ok, err = check_orders(cf, False)
    if not ok:
        print(err)
        # return ok, err
    ok, err = check_product_op_count(cf, False)
    if not ok:
        print(err)
        # return ok, err
    ok, err = check_notBefore(cf, False)
    if not ok:
        print(err)
        # return ok, err
    ok, err = check_op_time(cf, False)
    if not ok:
        print(err)
        # return ok, err
    ok, err = check_overlap_inside_product(cf, False)
    if not ok:
        print(err)
        # return ok, err

    ok, err = check_rc_usage(cf, False)
    if not ok:
        print(err)
        # return ok, err
    return True, 'OK'


def check_notBefore(cf, early_stop=False):
    errors = []
    for order, df_up_order in cf.df_up.groupby(["orderCode"]):
        boundary = cf.tab_time_constraint.get(order)
        notbefore = boundary.get("notBefore")
        # print(notbefore)
        if notbefore:
            start = df_up_order["start"].min()
            # print(start)
            if start < notbefore:
                msg = f"too early to start: {order}"
                if early_stop:
                    return False, msg
                errors.append(msg)

    if not errors:
        return True, "OK"
    return False, errors
