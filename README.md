## 需求：
* pandas
* numpy
* 下載資料中的 jobs.json 複製於執行目錄

## 用法：

`python3 validator.py [submit_file]`

submit_file: 未指定時，預設 submit.json

## 錯誤解說：
 * `submit column mismatch`: submit 格式錯誤
 * `submit find invalid columns`: submit 格式錯誤
 * `submit find 1613/1614 operations.`: 預期工序數不足，本次題目，答案需包含 1614 件工序。
 * `bad record`: jobs.json 裡找不到對應的工序
 * `not enough time`: 'D' 單位工時不足日
 * `invalid start/end time`: 指定時間非出勤時段
 * `allocate not enough time`: 指定時間段，工時不足
 * `worktime overlapped`: 前一工序尚未完工；舉例說明：生產數量5，seq:20 還只完成2件時，指派執行 seq:30 第3件
 * `run out of resource`: 資源使用量超過可使用總數
 * `usage less than`: 指定使用量超過單次下限
 * `usage larger than`: 指定使用量超過單次上限
 * `too early to start`: 開工時間早於 notBefore
