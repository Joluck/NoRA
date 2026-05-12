# 项目管理
```
doc/ # 所有项目的文档
perl/ # 包括所有TRL实现相关的东西
├── config.py # TRL实验的参数dataclass
├── data/
│   ├── openr1.py # openr1数据的处理函数和reward函数
│   ├── .....
│   ....
├── eval/ # TRL实验用的评估套件（以后会去掉）
│   ├── grader.py # 打分用的
│   ....
├── lora/ # 各种peft的trl专用实现
│   ├── milora.py # milora
│   ....
├── trainer/ # 暂时没用
├── utils/
├── eval.py # 会去掉的
├── test.py # 会去掉的
├── train.py # 启动的地方
verl/ # verl repo作为submodule
scripts/ # 所有.sh脚本
```