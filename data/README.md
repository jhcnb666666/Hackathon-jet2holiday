# 数据说明

本目录用于保存项目的数据来源、清洗结果和模拟测试数据。

## 目录规划

- `raw/`：原始公开数据或原始数据下载说明
- `processed/`：经过清洗的数据
- `sample/`：用于前端和后端测试的模拟数据

## 公开数据来源

数据集名称：Student Health Behavior and Academic Performance

来源：Florida Gulf Coast University Data Repository

官方网站：
https://dataverse.fgcu.edu/dataset.xhtml?persistentId=doi:10.60863/SF/HWI1MX

数据包含大学生的睡眠、运动、饮食、工作情况和自报 GPA。

## 使用限制

- 调查数据为学生自我报告。
- 数据之间的关联不能证明因果关系。
- 不同工作表没有可靠的统一学生编号，不能按照行号直接合并。
- 本项目不使用这些数据进行医学诊断。
- 不上传真实学生的姓名、学号、联系方式或心理健康记录。
- 模拟数据必须明确标注为“模拟演示数据”。
