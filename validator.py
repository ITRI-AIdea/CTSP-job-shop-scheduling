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

import sys
import json

import pandas as pd

import util as validator

pd.set_option("display.max_rows", None)


def load_submit(submit_name: str) -> pd.DataFrame:
    try:
        msg = f"failed to decode {submit_name}."
        with open(submit_name, "r") as fin:
            upload = json.load(fin)
        msg = f"{submit_name} find no solution element."
        upload = upload.get("solution")
        msg = f"{submit_name} can not convert to dataframe."
        return pd.DataFrame.from_dict(upload)
    except Exception as e:
        print(msg)
        print(str(e))


if __name__ == '__main__':
    ok, data_total = validator.import_data('jobs.json')
    if not ok:
        print("load environment setting failed.")
        sys.exit(-1)
    js = validator.JobShop(data_total)

    submit_name = sys.argv[1] if len(sys.argv) > 1 else "submit.json"
    df_up = load_submit(submit_name)

    ok, msg = validator.prepare(js, df_up)
    if not ok:
        print(msg)
        sys.exit(-1)
    ok, msg = validator.check(js)
    if not ok:
        print(msg)
    print('check ok.')
