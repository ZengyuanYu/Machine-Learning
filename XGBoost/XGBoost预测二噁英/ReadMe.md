## 二噁英数据处理
### 对1#锅炉进行处理

- 1.文件解释

	- **二噁英工控数据.xlsx** 为原始数据，里面包含很多sheet来存放不同的参数以及二噁英浓度排放指标；
	- **X_y.xlsx** 第一步处理得到的数据，将原始数据的所有表上面的参数存放到一个表当中来，其中将二噁英的浓度排放存入到最后一列；
	- **X_y.csv** 将excel文件里面的每一个小时的数据量进行合并并重写到此csv文件之中，按照时间排序，每一个小时出一组数据；
	- **X_y_12-04.csv** 由于前面的数据没有对应的二噁英排放浓度所以删除前面的行数；
	- **X_y_ereying_nozero.csv** 删除文件中二噁英为0的行；
	- **X_y_all_nozero.csv** 将参数中全部为零但是二噁英浓度不为零的行删除，得到的即有训练数据对应标签的数据。
- 2.调参记录

**调参图表**

|No.|learning_rate|n_estimators|max_depth|$R^2$
|---|:-----------:|:-----------:|:------:|:----:|
|1|0.1|500|16|0.729|
|2|0.1|500|6|0.789|

- 3.测试
	- 今天将数据进行分离，分为训练集和验证集。
