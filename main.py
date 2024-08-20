import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
import json
import os

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


def produce_distribution_plot(old_series:pd.Series, new_series:pd.Series, tag:str, measurement_type:str, plot_saving_directory:Path):
    print(f"This is only looking at script : '{tag}' and measurement : '{measurement_type}'")

    old_range_start, old_range_end = get_oulier_range(df_series=old_series,
                                                      min=0,
                                                      max=float('inf'))  # No max value (positive infinity)

    new_range_start, new_range_end = get_oulier_range(df_series=new_series,
                                                      min=0,
                                                      max=float('inf'))  # No max value (positive infinity)

    old_dist_saving_path = plot_saving_directory / f"old_dist_{tag}_{measurement_type}.jpg"
    old_dist_saving_path.parent.mkdir(parents=True, exist_ok=True)

    new_dist_saving_path = plot_saving_directory / f"new_dist_{tag}_{measurement_type}.jpg"
    new_dist_saving_path.parent.mkdir(parents=True, exist_ok=True)

    histogram_plot(data=old_series,
                              bins=100,
                              range=[old_range_start, old_range_end],
                              title=f"New: Distribution of {tag} - {measurement_type}",
                              plot_saving_path=old_dist_saving_path)
                              
    histogram_plot(data=new_series,
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


def aggregate_txt_report(tag:str, measurement_type:str, summary:str, old_df_count:int, old_outliers_count:int, new_df_count:int, new_outliers_count:int)->str:

    result = ""
    result += (f"Hypothesis Testing on {tag} - {measurement_type}. We will select alpha of 0.05.\n")
    result += f"OLD version has {old_df_count} data points with {old_outliers_count} outliers.\n"
    result += f"OLD version outliers percentage: {(old_outliers_count/old_df_count)*100}%\n"
    result += f"NEW version has {new_df_count} data points with {new_outliers_count} outliers.\n"
    result += f"NEW version outliers percentage: {(new_outliers_count/new_df_count)*100}%\n"
    result += (f"Null Hypothesis: The newer version has no significant difference in performance compared to the previous version.\n")
    result += (f"Alternative Hypothesis: The newer version has a significant difference in performance compared to the previous version.\n")
    result += ("\n")
    result += (summary+"\n")
    result += ("-"*50+"\n\n")

    return result


def construct_measurement_dict(measurement_type:str, tag:str, old_outliers_count:int, new_outliers_count:int, p_value:float, summary:str)->dict:

    report_measurement = dict()
    report_measurement["name"] = measurement_type
    report_measurement["old_version_distribution"] = f"old_dist_{tag}_{measurement_type}.jpg"
    report_measurement["new_version_distribution"] = f"new_dist_{tag}_{measurement_type}.jpg"
    report_measurement["old_version_qqplot"] = f"old_qq_{tag}_{measurement_type}.jpg"
    report_measurement["new_version_qqplot"] = f"new_qq_{tag}_{measurement_type}.jpg"
    report_measurement["old_version_outliers"] = old_outliers_count
    report_measurement["new_version_outliers"] = new_outliers_count
    report_measurement["hypothesis_test"] = f"p-value: {p_value}"
    report_measurement["conclusion"] = summary

    return report_measurement


def construct_tag_dict(tag:str, measurement_dicts:list)->dict:
    
    report_dict = dict()

    report_dict["name"] = tag
    report_dict["measurements"] = measurement_dicts

    return report_dict


def main():
    stats_df = get_statistic_df()  # This gets all the statistic from the given data file: mean, standard deviation, median, and more.

    stats_df.to_csv(Path(__file__).parent/"data"/"stats.csv", index=False)  # Save the statistics to a CSV file

    # # Read the data file into pandas and then plot
    old_df = pd.read_csv((Path(__file__).parent/"data"/"old.csv"))  # Old version data
    new_df = pd.read_csv((Path(__file__).parent/"data"/"new2.csv"))  # New version test data

    measurement_types_to_analyse = {"group_duration", "iteration_duration"}
    plot_saving_directory = Path(__file__).parent/"plot"
    json_report = dict()
    json_report["tags"] = list()
    txt_report = ""
    
    for tag in old_df["script"].unique():

        measurement_dicts = list()  # This will store the measurement dictionary for each tag

        for measurement_type in measurement_types_to_analyse:
            # Produce QQ-plot
            old_series = old_df[(old_df["script"]==tag) & (old_df["_measurement"]==measurement_type)]["_value"]
            old_range_start, old_range_end = get_oulier_range(df_series=old_series,
                                                            min=0,
                                                            max=float('inf'))  # No max value (positive infinity)
            old_series_no_outliers = old_series[(old_series >= old_range_start) & (old_series <= old_range_end)]
            old_outliers_count = old_series.shape[0] - old_series_no_outliers.shape[0]

            new_series = new_df[(new_df["script"]==tag) & (new_df["_measurement"]==measurement_type)]["_value"]
            new_range_start, new_range_end = get_oulier_range(df_series=new_series,
                                                            min=0,
                                                            max=float('inf'))  # No max value (positive infinity)
            new_series_no_outliers = new_series[(new_series >= new_range_start) & (new_series <= new_range_end)]
            new_outliers_count = new_series.shape[0] - new_series_no_outliers.shape[0]

            produce_distribution_plot(old_series=old_series_no_outliers,
                                    new_series=new_series_no_outliers,
                                    tag=tag,
                                    measurement_type=measurement_type,
                                    plot_saving_directory=plot_saving_directory)
            
            # Produce QQ-plot
            old_qq_plot_save_path = plot_saving_directory / f"old_qq_{tag}_{measurement_type}.jpg"
            old_qq_plot_save_path.parent.mkdir(parents=True, exist_ok=True)

            qq_plot(data=old_series_no_outliers,
                    dist="norm",
                    plot_saving_path=old_qq_plot_save_path)
            
            new_qq_plot_save_path = plot_saving_directory / f"new_qq_{tag}_{measurement_type}.jpg"
            new_qq_plot_save_path.parent.mkdir(parents=True, exist_ok=True)

            qq_plot(data=new_series_no_outliers,
                    dist="norm",
                    plot_saving_path=new_qq_plot_save_path)
            
            # Perform hypothesis testing + Add conclusion to report
            summary = ""
            _, p_value = stats.ttest_ind(a=old_series_no_outliers, b=new_series_no_outliers, equal_var=1)
            if p_value <= 0.05:
                if new_series_no_outliers.mean() < old_series_no_outliers.mean():
                    summary = f"P-Value: {p_value}. Base on our testing, the newer version has a FASTER performance than the previous version."

                else:
                    summary = f"P-Value: {p_value}. Base on our testing, the newer version has a SLOWER performance than the previous version."
                    
            else:
                summary = f"P-Value: {p_value}. The newer version has NO SIGNIFICANT DIFFERENCE in performance compared to the previous version. No improvement or degradation."
            
            # Using the conclusion, we will create a text that summarize the test by:
            # including the p_value, the number of outliers and its percentage, the conclusion that whether there exist a significant difference in the comparison
            # This is added to the overall text for writing into text file later
            txt_report += aggregate_txt_report(tag=tag,
                                measurement_type=measurement_type,
                                summary=summary,
                                old_df_count=old_series.shape[0],
                                old_outliers_count=old_outliers_count,
                                new_df_count=new_series.shape[0],
                                new_outliers_count=new_outliers_count)
            
            # Add the report of this tag and measurement type that is similar to the dictionary for json report
            measurement_dict = construct_measurement_dict(measurement_type=measurement_type,
                                                          tag=tag,
                                                          old_outliers_count=old_outliers_count,
                                                          new_outliers_count=new_outliers_count,
                                                          p_value=p_value,
                                                          summary=summary)
            measurement_dicts.append(measurement_dict)

        # Add the tag and its measurement dictionary to the json report
        tag_dict = construct_tag_dict(tag=tag,
                                      measurement_dicts=measurement_dicts)
        
        json_report["tags"].append(tag_dict)

    # Write the report to a text file and a json file
    with open(Path(__file__).parent/"report"/"report.txt", "w") as f:
        f.write(txt_report)

    with open(Path(__file__).parent/"report"/"report.json", "w") as f:
        json.dump(json_report, f, indent=4)

    # Generate HTML report
    generate_html_report(data=json_report,
                         output_path=Path(__file__).parent/"report"/"report.html")


def generate_html_report(data, output_path:Path):
    """
    :param data: Data to be rendered in the HTML report
        - data format:
        "tags": [
        {
            "name": "API Script 1",
            "measurements": [
                {
                    "name": "Measurement 1",
                    "old_version_distribution": "old_dist_plot1.png",
                    "new_version_distribution": "new_dist_plot1.png",
                    "old_version_qqplot": "old_qq_plot1.png",
                    "new_version_qqplot": "new_qq_plot1.png",
                    "old_version_outliers": 5,
                    "new_version_outliers": 3,
                    "hypothesis_test": "p-value: 0.05, t-statistic: 2.31",
                    "conclusion": "The new version shows a significant improvement."
                },
                {
                    "name": "Measurement 2",
                    "old_version_distribution": "old_dist_plot2.png",
                    "new_version_distribution": "new_dist_plot2.png",
                    "old_version_qqplot": "old_qq_plot2.png",
                    "new_version_qqplot": "new_qq_plot2.png",
                    "old_version_outliers": 8,
                    "new_version_outliers": 4,
                    "hypothesis_test": "p-value: 0.01, t-statistic: 3.47",
                    "conclusion": "The new version is statistically better."
                }
            ]
        },
        # More
    ]
    """

    # Set up Jinja2 environment and load template
    env = Environment(loader=FileSystemLoader(searchpath='./template'))
    template = env.get_template('report.html')

    # Render the template with the data
    html_content = template.render(tags=data["tags"])

    # Save the rendered HTML to a file
    with output_path.open(mode="w") as file:
        file.write(html_content)


if __name__ == "__main__":
    main()