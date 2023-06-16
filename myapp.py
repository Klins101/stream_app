import pandas as pd
import streamlit as st
from etna.datasets import TSDataset
from etna.models import CatBoostPerSegmentModel
from etna.transforms import (
    DensityOutliersTransform,
    TimeSeriesImputerTransform,
    LinearTrendTransform,
    LagTransform,
    DateFlagsTransform,
    FourierTransform,
    SegmentEncoderTransform,
    MeanTransform,
)
from etna.pipeline import Pipeline
from etna.analysis import plot_forecast
from etna.metrics import SMAPE
import sys,os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path,relative_path)

print("HI bibin")
st.title("Hi Bibin")
st.subheader("Welcome to Data Science")

data_df = pd.read_csv(resource_path("monthly-australian-wine-sales.csv"))

# st.title("My App - Test Task")
# st.subheader("The data - monthly-australian-wine-sales")
# Load data
# data_df = pd.read_csv("monthly-australian-wine-sales.csv")
data_df.columns = ["timestamp", "target"]
data_df["segment"] = "main"

# Create TSDataset
df = TSDataset.to_dataset(data_df)
ts = TSDataset(df, freq="MS")

# Display data
st.dataframe(data_df.head(10))
st.subheader("Choose Transforms & Plot Graph")

# Choose a horizon
HORIZON = 14

# Choose transforms
transforms_feature_options = {
    "LagTransform": LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
    "DateFlagsTransform": DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
    "FourierTransform": FourierTransform(period=360.25, order=6, out_column="fourier"),
    "MeanTransform": MeanTransform(in_column="target", window=12, seasonality=7),
}

selected_transforms_withfeature = st.multiselect("Select one or more Transforms with feature", list(transforms_feature_options.keys()))

transforms_with_feature = [transforms_feature_options[transform] for transform in selected_transforms_withfeature]
print(len(transforms_with_feature))
if(len(transforms_with_feature)==0):
    pass #Please select one or more transform 
else:
    # Choose transforms
    transforms_options = {
        "DensityOutliersTransform": DensityOutliersTransform(in_column="target", distance_coef=3.0),#nofeature
        "TimeSeriesImputerTransform": TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),#nofeature
        "LinearTrendTransform": LinearTrendTransform(in_column="target"),#nofeature
        "SegmentEncoderTransform": SegmentEncoderTransform(),#nofeature
        # "MeanTransform": MeanTransform(in_column="target", window=12, seasonality=7),
        # "LagTransform": LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
        # "DateFlagsTransform": DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
        # "FourierTransform": FourierTransform(period=360.25, order=6, out_column="fourier"),
    }

    selected_transforms_without_feature = st.multiselect("Add more Transforms", list(transforms_options.keys()))

    transforms_without_feature = [transforms_options[transform] for transform in selected_transforms_without_feature]
    print("%%%%%%%%",len(transforms_without_feature))
    # transforms_choice = lambda transforms: transforms_options["LagTransform"] if len(transforms)==0 else transforms
    # transforms_c = transforms_choice(transforms)

    if(len(transforms_without_feature)==0):
        transforms = transforms_with_feature
    else:
        transforms = transforms_with_feature + transforms_without_feature
   
   
    train_ts, test_ts = ts.train_test_split(test_size=HORIZON) # split  train/test dataset
    print(test_ts,test_ts) #printing and checking data

    # Prepare model,Create and fit the pipeline
    model = CatBoostPerSegmentModel()
    pipeline = Pipeline(model=model, transforms=transforms, horizon=HORIZON)
    pipeline.fit(train_ts)
    
  
    forecast_ts = pipeline.forecast()   # Make a forecast

    # Calculate metric
    metric = SMAPE(mode="macro")
    metric_value = metric(y_true=test_ts, y_pred=forecast_ts)

   
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning
    # Display forecast and metric
    st.subheader("Plot Forcast")
    st.pyplot(plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=50))
    st.subheader("Metric Result - SMAPE")
    st.metric(label="SMAPE", value=metric_value)    