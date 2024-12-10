import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# функция для удаления выбросов
def del_outlier(column, df):
    q_low = df[column].quantile(0.25)
    q_hi = df[column].quantile(0.75)
    q_range = q_hi - q_low
    df = df[
        (df[column] >= (q_low - 1.5 * q_range)) & (df[column] <= (q_hi + 1.5 * q_range))
    ]


# открываем скачанную таблицу
url = "telecom_users.csv"
df = pd.read_csv(url, na_values=" ", keep_default_na=True)

# смотрим типы данных столбцов
# print(df.dtypes)
# print(df.describe())

# проверка наличия нулевых значений в столбцах
df.columns[df.isna().any()].to_list()

# удаляем строки с пустыми значениями
df = df.dropna()
# удаляем столбцы, не влияющие на целевой показатель
df = df.drop(["Unnamed: 0", "customerID"], axis=1)

# обрабатываем выбросы
del_outlier("MonthlyCharges", df)
del_outlier("TotalCharges", df)
del_outlier("tenure", df)

# преобразуем столбцы типа object в int8 (кодируем категориальные данные)
cat_columns = df.select_dtypes(["object"]).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

# нормализуем и стандартизируем данные
scaler = MinMaxScaler()
df[["MonthlyCharges", "TotalCharges", "tenure"]] = scaler.fit_transform(
    df[["MonthlyCharges", "TotalCharges", "tenure"]]
)

scaler = StandardScaler()
df[["MonthlyCharges", "TotalCharges", "tenure"]] = scaler.fit_transform(
    df[["MonthlyCharges", "TotalCharges", "tenure"]]
)


# гистограммы числовым значений
numeric_features = ["MonthlyCharges", "TotalCharges", "tenure"]

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df[col]
    feature.hist(bins=100, ax=ax)
    ax.axvline(feature.mean(), color="red", linestyle="dashed", linewidth=2)
    ax.axvline(feature.median(), color="cyan", linestyle="dashed", linewidth=2)
    ax.set_title(col)
plt.show()
