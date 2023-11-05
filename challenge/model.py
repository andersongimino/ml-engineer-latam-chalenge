
import pandas as pd
from datetime import datetime
import xgboost as xgb
from typing import Tuple, Union, List, Any

class DelayModel:

    def __init__(self):
        self._model = xgb.XGBClassifier()
        self.features = None
        self.expected_columns = [
            "Fecha_I", "Vlo_I", "Ori_I", "Des_I", "Emp_I",
            "Fecha_O", "Vlo_O", "Ori_O", "Des_O", "Emp_O",
            "DIA", "MES", "AÑO", "DIANOM", "TIPOVUELO",
            "OPERA", "SIGLAORI", "SIGLADES"
        ]
        self.all_features = ['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada',
                             'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines',
                             'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways',
                             'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans',
                             'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA',
                             'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings',
                             'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas',
                             'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines',
                             'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_10', 'MES_11', 'MES_12',
                             'MES_2', 'MES_3', 'MES_4', 'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9']
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    @staticmethod
    def get_period_day(date_str: str) -> str:
        if date_str == "0" or date_str == 0:
            return 'night'
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()
        if datetime.strptime("05:00", '%H:%M').time() <= date_time < datetime.strptime("12:00", '%H:%M').time():
            return 'morning'
        elif datetime.strptime("12:00", '%H:%M').time() <= date_time < datetime.strptime("19:00", '%H:%M').time():
            return 'afternoon'
        else:
            return 'night'

    @staticmethod
    def is_high_season(date_str: str) -> int:
        if date_str == "0" or date_str == 0:
            return 0
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
        if fecha_o_str == 0 or fecha_i_str == 0 or fecha_o_str == "0" or fecha_i_str == "0":
            return 0.0
        fecha_o = datetime.strptime(fecha_o_str, '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(fecha_i_str, '%Y-%m-%d %H:%M:%S')
        return (fecha_o - fecha_i).total_seconds() / 60

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame],pd.DataFrame]:
        # Adicionar colunas faltantes com zeros
        if len(data.columns) < 18:
            missing_cols = set(self.expected_columns) - set(data.columns)
            for c in missing_cols:
                data[c] = 0
                data[c].astype(int)
        # print(data)
        data.rename(columns={'Fecha_I': 'Fecha-I'}, inplace=True)
        data.rename(columns={'Fecha_O': 'Fecha-O'}, inplace=True)
        data['MES'] = data['MES'].astype(str)

        #  Apply static methods as new columns
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(lambda row: self.get_min_diff(row['Fecha-O'], row['Fecha-I']), axis=1)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )

        if not isinstance(self.features, pd.DataFrame):
            self.features = pd.DataFrame(columns=features.columns)

        threshold_in_minutes = 15
        data['delay'] = (data['min_diff'] > threshold_in_minutes).astype(int)
        print(features.columns)
        if (len(features.columns) == 3):
            if self.features.empty:
                self.features = pd.DataFrame(columns=self.all_features)
            for col in features.columns:
                self.features[col] = self.features[col].add(features[col], fill_value=0).fillna(0)
            features = self.features.fillna(0)

        features = features[self.top_10_features]

        if target_column:
            target = data[[target_column]]
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame, col_target: str) -> None:
        self._model.fit(features, target[col_target])

    def predict(self, features: pd.DataFrame) -> Union[str, Any]:
        predictions = self._model.predict(features)
        return predictions.tolist()
