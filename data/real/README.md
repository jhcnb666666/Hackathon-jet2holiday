# 真实公开数据

本目录只包含从网上公开研究数据中整理出的真实记录，不包含模拟用户。

## 数据来源

- 数据集：Student Health Behavior and Academic Performance
- 发布机构：Florida Gulf Coast University Data Repository
- 官方网址：https://dataverse.fgcu.edu/dataset.xhtml?persistentId=doi:10.60863/SF/HWI1MX
- DOI：https://doi.org/10.60863/SF/HWI1MX
- 样本：614 名美国州立大学本科生
- 授权：CC0 1.0

## 文件

- `sleep_cleaned.csv`：睡眠时长、入睡时间、起床时间与 GPA
- `physical_activity_cleaned.csv`：运动与力量训练频率和 GPA
- `eating_habits_cleaned.csv`：饮食习惯和 GPA
- `work_cleaned.csv`：工作情况、每周工作时长和 GPA
- `demographics_cleaned.csv`：GPA、生理性别和年级

## 限制

- 数据来自匿名自报问卷，只能用于关联分析，不能证明因果关系。
- 各文件没有可验证的共同学生编号，禁止按 `record_id` 横向合并。
- 本数据不能用于医学或心理健康诊断。
- 清洗过程保留异常值，并通过质量标记进行说明。
