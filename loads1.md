::: {.cell .markdown}
Comparing building heating and cooling loads with respect to building
characteristics. The data contains 768 different building shapes and 8
characteristics. The goal is to use the 8 building features to predict
the heating and cooling load. The features are:

-   relative compactness
-   surface area - in meters squared
-   wall area - in meters squared
-   roof area - in meters squared
-   overall height - in meters
-   orientation - 2: North, 3: East, 4: South, 5: West
-   glazing area - as a percentage of the floor area - 0%, 10%, 25%, and
    40%
-   glazing area distribution:
    -   0: uniform - with 25% glazing on each side
    -   1: north - 55% on the north side and 15% on each of the other
        sides
    -   2: east - 55% on the east side and 15% on each of the other
        sides
    -   3: south - 55% on the south side and 15% on each of the other
        sides
    -   4: west - 55% on the west side and 15% on each of the other
        sides

The values to predict are the heating and cooling loads listed in
kWh/m2.

12 buildings were modeled with different shapes, surface areas and
dimensions. The overall volume was kept the same for each building at
772 meters squared. Considering twelve building forms with three glazing
area variations, six glazing area distributions each (5 distributions +
1 with no glazing), for four orientations, we obtain 12 × 3 × 6 × 4 =
768 different building variations.

The dataset, which can be found
[here](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) was
donated to the University of California at Irving.
:::

::: {.cell .code execution_count="1"}
``` {.python}
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
:::

::: {.cell .code execution_count="2"}
``` {.python}
data = pd.read_csv('Energy data.csv')
```
:::

::: {.cell .code execution_count="3"}
``` {.python}
data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
```
:::

::: {.cell .code execution_count="4"}
``` {.python}
data.head()
```

::: {.output .execute_result execution_count="4"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relative_compactness</th>
      <th>surface_area</th>
      <th>wall_area</th>
      <th>roof_area</th>
      <th>overall_height</th>
      <th>orientation</th>
      <th>glazing_area</th>
      <th>glazing_area_distribution</th>
      <th>heating_load</th>
      <th>cooling_load</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.55</td>
      <td>21.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.90</td>
      <td>563.5</td>
      <td>318.5</td>
      <td>122.50</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>20.84</td>
      <td>28.28</td>
    </tr>
  </tbody>
</table>
</div>
:::
:::

::: {.cell .code execution_count="5"}
``` {.python}
data.describe()
```

::: {.output .execute_result execution_count="5"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relative_compactness</th>
      <th>surface_area</th>
      <th>wall_area</th>
      <th>roof_area</th>
      <th>overall_height</th>
      <th>orientation</th>
      <th>glazing_area</th>
      <th>glazing_area_distribution</th>
      <th>heating_load</th>
      <th>cooling_load</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.00000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.00000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.764167</td>
      <td>671.708333</td>
      <td>318.500000</td>
      <td>176.604167</td>
      <td>5.25000</td>
      <td>3.500000</td>
      <td>0.234375</td>
      <td>2.81250</td>
      <td>22.307201</td>
      <td>24.587760</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.105777</td>
      <td>88.086116</td>
      <td>43.626481</td>
      <td>45.165950</td>
      <td>1.75114</td>
      <td>1.118763</td>
      <td>0.133221</td>
      <td>1.55096</td>
      <td>10.090196</td>
      <td>9.513306</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.620000</td>
      <td>514.500000</td>
      <td>245.000000</td>
      <td>110.250000</td>
      <td>3.50000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>6.010000</td>
      <td>10.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.682500</td>
      <td>606.375000</td>
      <td>294.000000</td>
      <td>140.875000</td>
      <td>3.50000</td>
      <td>2.750000</td>
      <td>0.100000</td>
      <td>1.75000</td>
      <td>12.992500</td>
      <td>15.620000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.750000</td>
      <td>673.750000</td>
      <td>318.500000</td>
      <td>183.750000</td>
      <td>5.25000</td>
      <td>3.500000</td>
      <td>0.250000</td>
      <td>3.00000</td>
      <td>18.950000</td>
      <td>22.080000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.830000</td>
      <td>741.125000</td>
      <td>343.000000</td>
      <td>220.500000</td>
      <td>7.00000</td>
      <td>4.250000</td>
      <td>0.400000</td>
      <td>4.00000</td>
      <td>31.667500</td>
      <td>33.132500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.980000</td>
      <td>808.500000</td>
      <td>416.500000</td>
      <td>220.500000</td>
      <td>7.00000</td>
      <td>5.000000</td>
      <td>0.400000</td>
      <td>5.00000</td>
      <td>43.100000</td>
      <td>48.030000</td>
    </tr>
  </tbody>
</table>
</div>
:::
:::

::: {.cell .markdown}
To simplify the model, we\'ll sum heating and cooling load into a new
column called \"energy load\". Then we\'ll eliminate the heating and
cooling load columns.
:::

::: {.cell .code execution_count="6"}
``` {.python}
data_new = data.copy()
```
:::

::: {.cell .code execution_count="7"}
``` {.python}
data_new['energy load'] = data['heating_load'] + data['cooling_load']
```
:::

::: {.cell .code execution_count="8"}
``` {.python}
data_new = data_new.drop(columns=['heating_load', 'cooling_load'])
```
:::

::: {.cell .markdown}
The new dataset:
:::

::: {.cell .code execution_count="9"}
``` {.python}
data_new.head(2)
```

::: {.output .execute_result execution_count="9"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relative_compactness</th>
      <th>surface_area</th>
      <th>wall_area</th>
      <th>roof_area</th>
      <th>overall_height</th>
      <th>orientation</th>
      <th>glazing_area</th>
      <th>glazing_area_distribution</th>
      <th>energy load</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>36.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.98</td>
      <td>514.5</td>
      <td>294.0</td>
      <td>110.25</td>
      <td>7.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>36.88</td>
    </tr>
  </tbody>
</table>
</div>
:::
:::

::: {.cell .markdown}
Correlation between variables
:::

::: {.cell .code execution_count="10"}
``` {.python}
# Setting the format for diplaying the correlation table
pd.set_option('display.float_format', lambda x: '{:,.3f}'.format(x))
```
:::

::: {.cell .code execution_count="11"}
``` {.python}
data_new.corr().sort_values('energy load', ascending=False)
```

::: {.output .execute_result execution_count="11"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relative_compactness</th>
      <th>surface_area</th>
      <th>wall_area</th>
      <th>roof_area</th>
      <th>overall_height</th>
      <th>orientation</th>
      <th>glazing_area</th>
      <th>glazing_area_distribution</th>
      <th>energy load</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>energy load</th>
      <td>0.632</td>
      <td>-0.669</td>
      <td>0.445</td>
      <td>-0.867</td>
      <td>0.898</td>
      <td>0.006</td>
      <td>0.241</td>
      <td>0.070</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>overall_height</th>
      <td>0.828</td>
      <td>-0.858</td>
      <td>0.281</td>
      <td>-0.973</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.898</td>
    </tr>
    <tr>
      <th>relative_compactness</th>
      <td>1.000</td>
      <td>-0.992</td>
      <td>-0.204</td>
      <td>-0.869</td>
      <td>0.828</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.632</td>
    </tr>
    <tr>
      <th>wall_area</th>
      <td>-0.204</td>
      <td>0.196</td>
      <td>1.000</td>
      <td>-0.292</td>
      <td>0.281</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>glazing_area</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.213</td>
      <td>0.241</td>
    </tr>
    <tr>
      <th>glazing_area_distribution</th>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.213</td>
      <td>1.000</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>orientation</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>surface_area</th>
      <td>-0.992</td>
      <td>1.000</td>
      <td>0.196</td>
      <td>0.881</td>
      <td>-0.858</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.669</td>
    </tr>
    <tr>
      <th>roof_area</th>
      <td>-0.869</td>
      <td>0.881</td>
      <td>-0.292</td>
      <td>1.000</td>
      <td>-0.973</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>-0.867</td>
    </tr>
  </tbody>
</table>
</div>
:::
:::

::: {.cell .code execution_count="12"}
``` {.python}
plt.figure(figsize = (12,12))
sns.heatmap(data_new.corr(), annot=True, linewidths=1, linecolor='blue');
```

::: {.output .display_data}
![](6fad61d7b2832982a6413e327c9b6044734cf298.png)
:::
:::

::: {.cell .markdown}
Correlation shows how features are related to each other using a linear
correlation between variables. Keep in mind that since it only find the
linear relationship between variables, it will not be able to find the
non-linear relationship. A better metric to use is called *feature
importance*, which we will determine after the model is built.
:::

::: {.cell .markdown}
On to the model
:::

::: {.cell .code execution_count="12"}
``` {.python}
y = data_new['energy load']
```
:::

::: {.cell .code execution_count="13"}
``` {.python}
X = data_new.drop(columns='energy load')
```
:::

::: {.cell .code execution_count="14"}
``` {.python}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
```
:::

::: {.cell .code execution_count="15"}
``` {.python}
model = RandomForestRegressor(n_estimators=100, random_state=23)
```
:::

::: {.cell .code execution_count="16"}
``` {.python}
model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="16"}
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
               oob_score=False, random_state=23, verbose=0, warm_start=False)
:::
:::

::: {.cell .code execution_count="17"}
``` {.python}
predictions = model.predict(X_test)
```
:::

::: {.cell .code execution_count="18"}
``` {.python}
from sklearn.metrics import r2_score
```
:::

::: {.cell .code execution_count="19"}
``` {.python}
r2_score(y_test, predictions)
```

::: {.output .execute_result execution_count="19"}
    0.9931211166410133
:::
:::

::: {.cell .markdown}
The random forest model provides us with a highly accurate R2 score of
0.993
:::

::: {.cell .code execution_count="20"}
``` {.python}
for feature in zip(data_new.columns, model.feature_importances_):
    print(feature)
```

::: {.output .stream .stdout}
    ('relative_compactness', 0.5060543116884478)
    ('surface_area', 0.14821753795820464)
    ('wall_area', 0.04233577705009435)
    ('roof_area', 0.08329432066422113)
    ('overall_height', 0.14625311504507843)
    ('orientation', 0.0023343330653874043)
    ('glazing_area', 0.06172701478673418)
    ('glazing_area_distribution', 0.009783589741831876)
:::
:::

::: {.cell .markdown}
After training the model, relative compactness is the most important
feature for predicting the heating and cooling load of a building. The
glazing area distribution is relatively unimportant.
:::

::: {.cell .markdown}
The higher the feature importance score, the more relevant the feature
is to the output variable, which is energy load in this case.
:::

::: {.cell .code execution_count="21"}
``` {.python}
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
```

::: {.output .display_data}
![](623e1422aea4e9851f6bb7e236a80d212af476de.png)
:::
:::

::: {.cell .code}
``` {.python}
```
:::
