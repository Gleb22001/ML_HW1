
import fastapi
import uvicorn
import pandas as pd
import joblib
import re
from typing import List, Optional
from starlette.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# Для работы с асинхронными операциями ввода-вывода (например, загрузка файлов)
import aiofiles

app = FastAPI()
# Описание базового объекта
model = joblib.load("final_model.pkl")

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

# Коллекция объектов
class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Подготовка данных для предсказания
    input_data = pd.DataFrame([[item.year, item.km_driven, item.mileage, item.engine, item.max_power, item.torque]], columns=['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque'])
    df_test=input_data
    df_test['mileage'] = df_test['mileage'].str[:-5].astype(float)
    df_test['engine'] = df_test['engine'].str[:-3].astype(float)
    df_test['max_power'] = df_test['max_power'].str[:-4]
    df_test['max_power']=df_test.loc[df_test['max_power']!='', ['max_power'] ].astype(float)
    df_test['kgm'] = df_test['torque'].str.contains('kgm', na=False).astype(int)
    df_test['kgm']=df_test['kgm'].apply(lambda x: 1 if x==0  else 9.8)
    df_test['torque'] = df_test['torque'].str.replace(',', '', regex=False)

    def replace_non_numeric(torque_str):
        if pd.isna(torque_str):
            return torque_str  # Оставляем NaN как есть
        return re.sub(r'[^0-9.]', 'a', torque_str)

    df_test['cleaned_torque'] = df_test['torque'].apply(replace_non_numeric)



    def extract_values(value_str):
        if pd.isna(value_str):
            return value_str  # О
        # Извлекаем все числа из строки
        numbers = re.findall(r'\d*\.\d+|\d+', value_str)
    
        if not numbers:
            return None, None

        # Преобразуем извлеченные числа в float
        numbers = list(map(float, numbers))

        # Первый элемент
        first_value = numbers[0]

        # Если есть только два значения, возвращаем их
        if len(numbers) == 2:
            second_value = numbers[1]
        # Если есть три значения, возвращаем среднее из второго и третьего
        elif len(numbers) == 3:
            second_value = (numbers[1] + numbers[2]) / 2
        else:
            second_value = None  # Если значений не хватает, возвращаем None

        return first_value, second_value

    print(df_test['cleaned_torque'].apply(extract_values).apply(pd.Series))
    df_test[['torque', 'max_torque_rpm']] = df_test['cleaned_torque'].apply(extract_values).apply(pd.Series)
    df_test.drop('cleaned_torque', axis=1, inplace=True)
    df_test['torque']=df_test['torque']*df_test['kgm']
    df_test.drop('kgm', axis=1, inplace=True)

    X_test = df_test.loc[:, ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque','max_torque_rpm']]

    X_test['year2']=X_test['year']**2


    prediction = model.predict(X_test)
    return prediction[0]  # Возвращаем предсказанную цену

#@app.post("/predict_items")
#def predict_items(items: Items) -> List[float]:
#    predictions = []
##    for item in items.objects:
#        input_data = np.array([[item.year, item.km_driven]])
 #       prediction = model.predict(input_data)
 #       predictions.append(prediction[0])  # Добавляем предсказанную цену в список
 #   return predictions

@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    # Считываем CSV-файл
    df = pd.read_csv(file.file)
    df_test=df

    df_test['mileage'] = df_test['mileage'].str[:-5].astype(float)
    df_test['engine'] = df_test['engine'].str[:-3].astype(float)
    df_test['max_power'] = df_test['max_power'].str[:-4]
    df_test['max_power']=df_test.loc[df_test['max_power']!='', ['max_power'] ].astype(float)
    df_test['kgm'] = df_test['torque'].str.contains('kgm', na=False).astype(int)
    df_test['kgm']=df_test['kgm'].apply(lambda x: 1 if x==0  else 9.8)
    df_test['torque'] = df_test['torque'].str.replace(',', '', regex=False)

    def replace_non_numeric(torque_str):
        if pd.isna(torque_str):
            return torque_str  # Оставляем NaN как есть
        return re.sub(r'[^0-9.]', 'a', torque_str)

    df_test['cleaned_torque'] = df_test['torque'].apply(replace_non_numeric)



    def extract_values(value_str):
        if pd.isna(value_str):
            return value_str  # О
        # Извлекаем все числа из строки
        numbers = re.findall(r'\d*\.\d+|\d+', value_str)
    
        if not numbers:
            return None, None

        # Преобразуем извлеченные числа в float
        numbers = list(map(float, numbers))

        # Первый элемент
        first_value = numbers[0]

        # Если есть только два значения, возвращаем их
        if len(numbers) == 2:
            second_value = numbers[1]
        # Если есть три значения, возвращаем среднее из второго и третьего
        elif len(numbers) == 3:
            second_value = (numbers[1] + numbers[2]) / 2
        else:
            second_value = None  # Если значений не хватает, возвращаем None

        return first_value, second_value


    df_test[['torque', 'max_torque_rpm']] = df_test['cleaned_torque'].apply(extract_values).apply(pd.Series)
    df_test.drop('cleaned_torque', axis=1, inplace=True)
    df_test['torque']=df_test['torque']*df_test['kgm']
    df_test.drop('kgm', axis=1, inplace=True)

    df_test[df_test.select_dtypes(include='number').columns] = df_test.select_dtypes(include='number').fillna(df_test[df_test.select_dtypes(include='number').columns].median())
    #medians = df_train[df_train.select_dtypes(include='number').columns].median()
    #df_test[df_test.select_dtypes(include='number').columns] = df_test.select_dtypes(include='number').fillna(medians)

    X_test = df_test.loc[:, ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque','max_torque_rpm']]

    X_test['year2']=X_test['year']**2


#    # Проверяем наличие необходимых столбцов
#    if not set(['year', 'km_driven']).issubset(df.columns):
#        return {"error": "CSV must contain year and km_driven columns."}

    # Предсказания
    predictions = model.predict(X_test)
    
    # Создаем новый DataFrame для результатов
    df['predicted_price'] = predictions
    
    # Сохраняем результаты в временный CSV файл
    output_file = 'predicted_output.csv'
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, media_type='text/csv', filename=output_file)