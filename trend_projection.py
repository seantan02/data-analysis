import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from jinja2 import Environment, FileSystemLoader
from typing import Union
import math
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
import argparse

from config import trend_projection_summary_files, trend_projection_file_timestamps


def construct_trend_data_dict(file_directory:Path, summary_files:list, timestamps:list):
    df = None

    trend_data_dict = dict()
    trend_data_dict

    # Now we go through each summary file and create a dictionary of data to be used for trend projection
    for i, summary_file in enumerate(summary_files):
        df = pd.read_csv(file_directory / summary_file)

        # We want to make sure the column names are consistent and easy to work with
        new_column_names = dict()

        for j, col in enumerate(df.columns):
            if j == 0:
                new_column_names[col] = "script"
                continue

            if "group" in col.lower() and "duration" in col.lower():
                if "90" in col:
                    new_column_names[col] = "group_duration_90"
                elif "95" in col:
                    new_column_names[col] = "group_duration_95"
                elif "99" in col:
                    new_column_names[col] = "group_duration_99"
                continue

            if "http_req_duration" in col.lower():
                if "95" in col:
                    new_column_names[col] = "http_req_duration_95"
                continue
            
            new_column_names[col] = col

        df = df.rename(columns=new_column_names)  # Update df column names

        for row in df.itertuples():
            if any(pd.isna(value) for value in row):
                continue

            if row.script not in trend_data_dict:
                trend_data_dict[row.script] = dict()
                trend_data_dict[row.script]["timestamp"] = list()
                trend_data_dict[row.script]["group_duration_90"] = list()
                trend_data_dict[row.script]["group_duration_95"] = list()
                trend_data_dict[row.script]["group_duration_99"] = list()
                trend_data_dict[row.script]["http_req_duration_95"] = list()
            
            if (hasattr(row, "group_duration_90") and float(row.group_duration_90) == 0.00) or (hasattr(row, "group_duration_95") and float(row.group_duration_95) == 0.00)\
               or (hasattr(row, "group_duration_99") and float(row.group_duration_99) == 0.00) or (hasattr(row, "http_req_duration_95") and float(row.http_req_duration_95) == 0.00):
                    continue
            
            trend_data_dict[row.script]["timestamp"].append(timestamps[i])
            
            if hasattr(row, "group_duration_90"):
                trend_data_dict[row.script]["group_duration_90"].append(row.group_duration_90)
            if hasattr(row, "group_duration_95"):
                trend_data_dict[row.script]["group_duration_95"].append(row.group_duration_95)
            if hasattr(row, "group_duration_99"):
                trend_data_dict[row.script]["group_duration_99"].append(row.group_duration_99)
            if hasattr(row, "http_req_duration_95"):
                trend_data_dict[row.script]["http_req_duration_95"].append(row.http_req_duration_95)
    print(f"Trend data dict: {trend_data_dict}")
    return trend_data_dict


def plot_trend_projection(original_x:list, original_y:list, x:list, y:list, label:str, plot_save_path:Path, 
                          title:str, x_label:str="Data", y_label:str="y", x_scale_function=None, x_ais_is_date=True):

    # Plot the results
    if x_scale_function is not None:
        original_x = x_scale_function(original_x)
        x = x_scale_function(x)

    plt.figure(figsize=(10, 6))
    plt.scatter(original_x, original_y, color='black', label='Original Data Points')
    plt.plot(x, y, color='blue', label=label)

    if x_ais_is_date:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.legend()
    plt.grid(True)  # Optional: Add grid for better readability

    # Save plot
    plt.savefig(plot_save_path)
    plt.close()


def loo_mse_cross_validation(model:Union[LinearRegression, Lasso], x:Union[np.ndarray, list], y:Union[np.ndarray, list], negative_exp=False) -> np.ndarray:
    loo = LeaveOneOut()

    mse_scores = []

    if isinstance(x, list):
        x = np.array(x).reshape(-1, 1)
    
    if isinstance(y, list):
        y = np.array(y).reshape(-1, 1)

    for train_index, test_index in loo.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if negative_exp:
            y_pred = np.exp(y_pred)
            y_test = np.exp(y_test)

        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return mse_scores


def produce_polynomial_regression_plot(degree, x:list, y:list, future_x:list, plot_save_path:Path, y_label:str="y", x_scale_function=None, min_y=50):
    # Fit polynomial and get coefficients
    # Convert datetime to numeric feature
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(future_x, list):
        future_x = np.array(future_x)

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    polyregression_x_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # Create and fit the model
    model = LinearRegression()

    model.fit(polyregression_x_poly, y)

    # Combine training and future data for plotting
    combined_timestamps = np.concatenate((x, future_x), axis=0)
    combined_x = np.array(combined_timestamps).reshape(-1, 1)
    combined_x_poly = poly_features.transform(combined_x)
    combined_y = model.predict(combined_x_poly)
    combined_y[combined_y < min_y] = min_y  # Cap the value to 50ms

    plot_trend_projection(original_x=x,
                          original_y=y,
                          x=combined_x,
                          y=combined_y,
                          label=f'Polynomial Regression (degree={degree})',
                          plot_save_path=plot_save_path,
                          title='Polynomial Regression with Monthly Interval Datetime',
                          x_label='Date',
                          y_label=y_label,
                          x_scale_function=x_scale_function)


def produce_log_regression_plot(x:list, y:list, future_x:list, plot_save_path:Path, y_label:str="y", x_scale_function=None, min_y=50):

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(future_x, list):
        future_x = np.array(future_x)

    # Create polynomial features
    log_transformer = FunctionTransformer(np.log)
    # Since y = a * e^-x + c, which is equivalent to log(y) = log(a) * -x + log(c)
    transformed_np_y = log_transformer.transform(y)

    # Create and fit the model
    model = LinearRegression()

    model.fit((x.reshape(-1, 1)), transformed_np_y)

    # Combine training and future data for plotting
    combined_timestamps = np.concatenate((x, future_x), axis=0)
    combined_x = np.array(combined_timestamps).reshape(-1, 1)
    # We want to plot y = a * exp(b*x) + c, logarithm the equation gives us
    # log(y) = log(a) + b*x + log(c)
    # Thus we can just transform y -> log(y) and it will be a linear regression
    combined_y = model.predict(combined_x)  # Now we have to exp() the result to get the actual y
    actual_combined_y = np.exp(combined_y)
    actual_combined_y[actual_combined_y < min_y] = min_y # Cap the value to 50ms

    plot_trend_projection(original_x=x,
                          original_y=y,
                          x=combined_x,
                          y=actual_combined_y,
                          label='Negative Exponential Regression',
                          plot_save_path=plot_save_path,
                          title='Negative Exponential Regression with Monthly Interval Datetime',
                          x_label='Date',
                          y_label=y_label,
                          x_scale_function=x_scale_function)
    

def produce_regression_plot(regression_type:str, x, y, future_x, plot_save_path, y_label, x_scale_function=None, min_y=50):

    if "polyreg_" in regression_type:
        degree = int(regression_type.split("_")[1])
        produce_polynomial_regression_plot(degree=degree,
                                            x=x,
                                            y=y,
                                            future_x=future_x,
                                            plot_save_path=plot_save_path,
                                            y_label=y_label,
                                            x_scale_function=x_scale_function,
                                            min_y=min_y)
    elif "negexp" in regression_type:
        produce_log_regression_plot(x=x,
                                    y=y,
                                    future_x=future_x,
                                    plot_save_path=plot_save_path,
                                    y_label=y_label,
                                    x_scale_function=x_scale_function,
                                    min_y=min_y)



def construct_model_valuation(polynomial_degrees:list[int], x:Union[np.ndarray, list], y:Union[np.ndarray, list]):
    
    np_x = x
    if isinstance(x, list):
        np_x = np.array(x).reshape(-1, 1)

    result = dict()
    result["model"] = list()

    # Loop through the polynomial degrees and perform n fold cross validation
    for degree in polynomial_degrees:

        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        transformed_x = poly_features.fit_transform(np_x)

        # Create and fit the model
        model = Lasso(alpha=0.25, max_iter=25000, tol=0.85)

        model_result = dict()
        model_result["name"] = f"Polynomial Regression (degree={degree})"
        model_result["acronym"] = f"polyreg_{degree}"
        model_result["mse_score"] = loo_mse_cross_validation(model=model,
                                                        x=transformed_x,
                                                        y=y)

        result["model"].append(model_result)


    # Negative Exponential Regression cross validation
    # Create the data transformer
    log_transformer = FunctionTransformer(np.log)
    transformed_np_y = log_transformer.transform(y)

    # Create and fit the model
    model = LinearRegression()

    model_result = dict()
    model_result["name"] = "Negative Exponential Regression"
    model_result["acronym"] = "negexp"
    model_result["mse_score"] = loo_mse_cross_validation(model=model,
                                                        x=np_x,
                                                        y=transformed_np_y,
                                                        negative_exp=True)

    result["model"].append(model_result)

    return result


def find_best_model(polynomial_degrees:list[int], x:Union[np.ndarray, list], y:Union[np.ndarray, list]):
    valuation_dict = construct_model_valuation(polynomial_degrees=polynomial_degrees,
                                               x=x,
                                               y=y)
    
    best_average_mse = 0
    best_model = None

    for model in valuation_dict["model"]:
        average_mse = np.mean(model["mse_score"])
        if best_model is None or average_mse < best_average_mse:
            best_average_mse = average_mse
            best_model = model["acronym"]

    return best_average_mse, best_model


def construct_trend_data_measurement_dict(name:str, plot_path:Path, mse:float)->dict:
    measurement_dict = dict()
    measurement_dict["name"] = name
    measurement_dict["plot"] = plot_path
    measurement_dict["mse"] = mse
    measurement_dict["me"] = math.sqrt(mse)

    return measurement_dict
    

def generate_trend_projection_html(title:str, data:dict, output_path:Path=(Path(__file__).parent/"report"/"trend_projection.html"), main_css_path:Path=(Path(__file__).parent/"static"/"trend_projection.css")):
    # Load the template
    env = Environment(loader=FileSystemLoader('./static'))
    template = env.get_template('trend_projection.html')

    # Render the HTML with the data
    html_output = template.render(title=title,
                                  main_css_file_path=main_css_path,
                                  report_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                                  HTML_trends_data=data)

    # Save the HTML to a file
    with output_path.open(mode="w") as f:
        f.write(html_output)


def generate_trend_projection():
    # Create the parser
    parser = argparse.ArgumentParser(description='Generate trend projection reports from the summary files')
    # Add the arguments
    parser.add_argument('--report_title', default="API Version Comparison", type=str, help='The title of the HTML report')
    # Parse the arguments
    args = parser.parse_args()

    # Step 1 read the files and for each API, append the data to a list
    
    file_directory = Path(__file__).parent / "data" / "trend_projection"

    trend_data_dict = construct_trend_data_dict(file_directory=file_directory,
                                                summary_files=trend_projection_summary_files,
                                                timestamps=trend_projection_file_timestamps)

    # Loop through the APIs and fit the Prophet model for each one
    plot_save_path = Path(__file__).parent / "plot" / "trend_projection"

    HTML_trends_data = dict()

    for key in trend_data_dict:
        api_data = trend_data_dict[key]
        api_timestamps = api_data["timestamp"]

        if len(api_timestamps) < 2:  # Since all the len of api_data should be the same, we can just check one of them. We need at least 2 data points
            print(f"API '{key}' does not have enough data points for trend projection. Miniumum of 2 data points are required but only {len(api_timestamps)} data points are available.")
            continue

        # Initialize the dictionary we will be using to generate HTML report
        if key not in HTML_trends_data:
            HTML_trends_data[key] = dict()
            HTML_trends_data[key]["measurement"] = list()

        measurements = ["group_duration_90", "group_duration_95", "group_duration_99", "http_req_duration_95"]
        y_labels = ["Group Duration 90th Percentile (ms)", "Group Duration 95th Percentile (ms)", "Group Duration 99th Percentile (ms)", "HTTP Request Duration 95th Percentile (ms)"]
        measurement_names = ["Group Duration 90th Percentile", "Group Duration 95th Percentile", "Group Duration 99th Percentile", "HTTP Request Duration 95th Percentile"]

        for i, measurement in enumerate(measurements):
            api_measurement_data = api_data[measurement]
            
            if len(api_measurement_data) < 2:  # Skip if there's less than 2 data points because we need at least 2
                continue

            timestamps_as_input = [(timestamp.year*12 + timestamp.month) for timestamp in api_timestamps]

            future_timestamps = []
            curr_timestamp = api_timestamps[-1]
            for _ in range(24):  # 2 years (24 months)
                curr_timestamp = curr_timestamp + relativedelta(months=1)
                future_timestamps.append(curr_timestamp)
            future_timestamps_as_input = [(timestamp.year*12 + timestamp.month) for timestamp in future_timestamps]

            # We will find the best fitted model for each of the data type

            best_mse, best_model = find_best_model(polynomial_degrees=[1, 2, 3, 4, 5],
                                                        x=timestamps_as_input,
                                                        y=api_measurement_data)
    
            # Produce the trend projection plot for Group Duration 90th Percentile
            def numeric_to_date_scale(numbers:Union[np.ndarray, list])->list:
                result = []
                if isinstance(numbers, np.ndarray):
                    if numbers.ndim==2 and numbers.shape[1] == 1:
                        numbers = numbers.squeeze(1).tolist()
                    else:
                        numbers = numbers.tolist()
                
                for number in numbers:
                    if number%12 == 0:
                        year = int(number/12) - 1
                        month = 12
                    else:
                        year = int(math.floor(number/12))
                        month = number%12

                    result.append(datetime.datetime(year=year, month=month, day=1))

                return result


            produce_regression_plot(regression_type=best_model,
                                    x=timestamps_as_input,
                                    y=api_measurement_data,
                                    future_x=future_timestamps_as_input,
                                    plot_save_path=(plot_save_path/f'{measurement}_{key}.jpg'),
                                    y_label=y_labels[i],
                                    x_scale_function=numeric_to_date_scale,
                                    min_y=50)
        
        

            measurement_dict = construct_trend_data_measurement_dict(name=measurement_names[i],
                                                                     plot_path=(plot_save_path/f'{measurement}_{key}.jpg'),
                                                                     mse=best_mse)
            HTML_trends_data[key]["measurement"].append(measurement_dict)  # Add the measurement trend projecton to the dictionary

    generate_trend_projection_html(title=args.report_title,
                                   data=HTML_trends_data,
                                   output_path=(Path(__file__).parent/"report"/"trend_projection.html"))  # Generate HTML report for trend projection
                                    
if __name__ == "__main__":
    generate_trend_projection()