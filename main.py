import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def find_best_distribution(data, distributions_to_check=['norm', 'lognorm', 'expon', 'gamma', 'beta', 'rayleigh', 'pareto']):
    """
    Find the best distribution that fits the data.
    
    :param data: Your dataset
    :param distributions_to_check: List of distribution names to check
    :return: DataFrame with distribution names and their respective p-values
    """
    results = []
    
    for dist_name in distributions_to_check:
        dist = getattr(stats, dist_name)
        
        # Fit distribution parameters
        params = dist.fit(data)
        
        # Perform Kolmogorov-Smirnov test
        D, p_value = stats.kstest(data, dist_name, args=params)
        
        results.append((dist_name, p_value))
    
    # Sort results by p-value in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return pd.DataFrame(results, columns=['Distribution', 'p-value'])


def get_oulier_range(df_series: pd.Series, min=0, max = 9999)->tuple:

    q1 = df_series.quantile(0.25)  # Get the 25th percentile
    q3 = df_series.quantile(0.75)  # Get the 75th percentile
    iqr = q3 - q1  # Get the interquartile range

    range_start = q1-(1.5*iqr)
    range_start = min if range_start < min else range_start  # If the range start is less than the minimum value, set it to the minimum value
    range_end = q3+(1.5*iqr)
    range_end = max if range_end > max else range_end  # If the range end is greater than the maximum value, set it to the maximum value

    return range_start, range_end


def histogram_plot(data, bins:int, range:list, title, plot_saving_path:Path):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, range=range)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(plot_saving_path)
    plt.close()


def qq_plot(data, dist, plot_saving_path:Path):
    plt.figure(figsize=(10, 6))
    plt.title(f"New: QQ-Plot")
    stats.probplot(data, dist=dist, sparams=(1,), plot=plt)
    plt.savefig(plot_saving_path)
    plt.close()


def produce_distribution_plot(old_group_duration:pd.Series, new_group_duration:pd.Series, tag:str, measurement_type:str, plot_saving_directory:Path):
    print(f"This is only looking at script : '{tag}' and measurement : '{measurement_type}'")
    print(f"First 5 entries of the value for old version: {old_group_duration.head(5)}")
    print(f"First 5 entries of the value for new version: {new_group_duration.head(5)}")

    old_range_start, old_range_end = get_oulier_range(df_series=old_group_duration,
                                                      min=0,
                                                      max=float('inf'))  # No max value (positive infinity)

    new_range_start, new_range_end = get_oulier_range(df_series=new_group_duration,
                                                      min=0,
                                                      max=float('inf'))  # No max value (positive infinity)

    x = int(len(os.listdir(plot_saving_directory)) / 2)

    old_dist_saving_path = plot_saving_directory / f"old_dist_{tag}_{measurement_type}_{x}.jpg"
    new_dist_saving_path = plot_saving_directory / f"new_dist_{tag}_{measurement_type}_{x}.jpg"

    histogram_plot(data=old_group_duration,
                              bins=100,
                              range=[old_range_start, old_range_end],
                              title=f"New: Distribution of {tag} - {measurement_type}",
                              plot_saving_path=old_dist_saving_path)
                              
    histogram_plot(data=new_group_duration,
                              bins=100,
                              range=[new_range_start, new_range_end],
                              title=f"New: Distribution of {tag} - {measurement_type}",
                              plot_saving_path=new_dist_saving_path)

def calculate_stats(group):
        return pd.Series({
            'mean': group['_value'].mean(),
            'median': group['_value'].median(),
            'mode': stats.mode(group['_value'])[0],
            'std_dev': group['_value'].std(),
            'percentile_25': np.percentile(group['_value'], 25),
            'percentile_50': np.percentile(group['_value'], 50),
            'percentile_75': np.percentile(group['_value'], 75),
            'percentile_90': np.percentile(group['_value'], 90),
            'percentile_95': np.percentile(group['_value'], 95),
            'percentile_99': np.percentile(group['_value'], 99),
            'count': len(group)
        })

def get_statistic_df(data_filepath=(Path(__file__).parent / "data" / "data.csv"), groups = ['script', '_measurement']) -> pd.DataFrame:
    # Read the CSV file
    df = pd.read_csv(data_filepath)

    # Group the data by tag and measurement_type
    grouped = df.groupby(groups)

    # Apply the function to each group
    results = grouped.apply(calculate_stats)

    # Reset index to make tag and measurement_type columns
    results = results.reset_index()

    return results


def main():
    # stats_df = get_statistic_df()  # This gets all the statistic from the given data file: mean, standard deviation, median, and more.
    
    # # Read the data file into pandas and then plot
    old_df = pd.read_csv((Path(__file__).parent/"data"/"old.csv"))  # Old version data
    new_df = pd.read_csv((Path(__file__).parent/"data"/"new.csv"))  # New version test data

    # measurement_types_to_analyse = {"group_duration", "iteration_duration"}
    # plot_saving_directory = Path(__file__).parent/"plot"
    # for measurement_type in measurement_types_to_analyse:

    #     for tag in old_df["script"].unique():

    #         produce_distribution_plot(old_df=old_df,
    #                                 new_df=new_df,
    #                                 tag=tag,
    #                                 measurement_type=measurement_type,
    #                                 plot_saving_directory=plot_saving_directory)

    # Produce QQ-plot
    old_group_duration = old_df[(old_df["script"]=="Add Appeal API") & (old_df["_measurement"]=="group_duration")]["_value"]
    old_range_start, old_range_end = get_oulier_range(df_series=old_group_duration,
                                                      min=0,
                                                      max=float('inf'))  # No max value (positive infinity)
    old_group_duration = old_group_duration[(old_group_duration >= old_range_start) & (old_group_duration <= old_range_end)]

    qq_plot(data=old_group_duration, dist="norm", plot_saving_path=Path(__file__).parent/"plot"/"qq_plot.jpg")

    best_distributions = find_best_distribution(old_group_duration)
    print(best_distributions)

    # If you want to see only distributions with p-value > 0.05
    best_fit = best_distributions.loc[best_distributions["p-value"].idxmax()]
    print("Distributions that fit the best:")
    print(best_fit)

    stat, p_value = stats.shapiro(old_group_duration)
    print(f"Shapiro-Wilk Test: Stat = {stat}, p-value = {p_value}")


if __name__ == "__main__":
    main()