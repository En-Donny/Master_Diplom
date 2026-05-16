import pandas as pd
import numpy as np
from model_PINN.preprocesser import *
from model_PINN.pinn import *


class Inference_PINN:
    def __init__(self, model_path:str = None):
        self.model_path = model_path if model_path else "./model_PINN/model_weights/pinn_run_new_01_05_26_model.pth"
        # создайте модель с теми же параметрами архитектуры, что и при обучении
        self.model = PINNNet(in_dim=6)
        map_location = torch.device('cpu')
        state = torch.load(self.model_path, map_location=map_location)
        self.model.load_state_dict(state)

    def predict(self, df):
        preprocess = Preprocesser()
        prep_df = preprocess.preproc(df)

        df_preds = predict_and_add_column(
            df_lab=prep_df,
            feature_cols=["I", "Q", "n", "x_1", "x_2", "x_3"],
            model=self.model,
            model_path=self.model_path,
            device='cpu',
            batch_size=1024,
            output_col='rho_pred_model'
        )
        return df_preds[["rho_pred_model"]].tail(1)
