
import pandas as pd
from datetime import datetime
import xgboost as xgb
from typing import Tuple, Union, List, Any


class DelayModel:

    def __init__(self):
        self._model = xgb.XGBClassifier()
        self.features = None

    @staticmethod
    def get_period_day(date_str: str) -> str:
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()
        if datetime.strptime("05:00", '%H:%M').time() <= date_time < datetime.strptime("12:00", '%H:%M').time():
            return 'morning'
        elif datetime.strptime("12:00", '%H:%M').time() <= date_time < datetime.strptime("19:00", '%H:%M').time():
            return 'afternoon'
        else:
            return 'night'

    @staticmethod
    def is_high_season(date_str: str) -> int:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        year = date_obj.year
        if any([
            datetime(year, 12, 15) <= date_obj <= datetime(year, 12, 31),
            datetime(year, 1, 1) <= date_obj <= datetime(year, 3, 3),
            datetime(year, 7, 15) <= date_obj <= datetime(year, 7, 31),
            datetime(year, 9, 11) <= date_obj <= datetime(year, 9, 30),
        ]):
            return 1
        else:
            return 0

    @staticmethod
    def get_min_diff(fecha_o_str: str, fecha_i_str: str) -> float:
        fecha_o = datetime.strptime(fecha_o_str, '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(fecha_i_str, '%Y-%m-%d %H:%M:%S')
        return (fecha_o - fecha_i).total_seconds() / 60

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data.rename(columns={'Fecha_I': 'Fecha-I'}, inplace=True)
        data.rename(columns={'Fecha_O': 'Fecha-O'}, inplace=True)
        # Aplicar os métodos estáticos como novas colunas
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(lambda row: self.get_min_diff(row['Fecha-O'], row['Fecha-I']), axis=1)
        # print("Dataframe")
        # print(data)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )
        # print("Features")
        # print(features)
        # print("self.Features")
        # print(self.features)
        threshold_in_minutes = 15
        data['delay'] = (data['min_diff'] > threshold_in_minutes).astype(int)

        if isinstance(self.features, pd.DataFrame):  # Verifica se é um DataFrame
            if len(self.features.columns) == 37:
                print("self.features existe nao precisa popular")
            else:
                print("self.features nao existe e precisa popular")
                self.features = features.assign(**{col: 0 for col in features.columns})
        else:
            print("self.features nao existe e precisa popular")
            self.features = pd.DataFrame(columns=features.columns)

        # print(self.features)

        if target_column:
            target = data[target_column]
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> str | Any:
        # Concatenamos os DataFrames
        df_concatenado = pd.concat([self.features, features], ignore_index=True)
        # Agora, agrupamos por índice (já que queremos a soma dos valores para cada coluna) e somamos os valores
        df_somado = df_concatenado.groupby(df_concatenado.index).sum()
        predictions = self._model.predict(df_somado)
        return predictions.tolist()
