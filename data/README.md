# 数据说明

本目录将真实公开数据、模拟演示数据和项目生成文档严格分开，避免混用。

## 目录结构

- `real/`：来源于网上公开研究的真实数据及来源说明
- `simulated/`：仅用于前端、后端和 Agent 流程测试的模拟数据
- `docs/`：字段字典、数据质量报告等项目生成文档

## 真实数据来源

数据集名称：Student Health Behavior and Academic Performance  
来源：Florida Gulf Coast University Data Repository  
官方网站：https://dataverse.fgcu.edu/dataset.xhtml?persistentId=doi:10.60863/SF/HWI1MX  
DOI：https://doi.org/10.60863/SF/HWI1MX  
授权：CC0 1.0

真实数据包含大学生的睡眠、运动、饮食、工作情况和自报 GPA。

## 重要说明

- `real/`中的数据来自公开研究，不包含本项目虚构的学生记录。
- `simulated/`中的记录全部是模拟数据，不能用于证明真实规律或产品效果。
- `docs/`中的文件由项目根据数据生成，不是独立调查数据。
- 公开调查数据为学生自我报告，变量关联不能证明因果关系。
- 各真实数据文件没有可验证的共同学生编号，禁止按照 `record_id` 横向合并。
- 本项目不使用这些数据进行医学或心理健康诊断。
- 不上传真实学生的姓名、学号、联系方式或个人心理健康记录。
