### Data Analysis

This project aims to produce reports of data analysis. 

This project will be able to a HTML report that contains distribution plots, QQ-plot, T-Test if distribution is normal, and permutation test with the null hypothesis of "There is no significant difference between old and new data."

This project also contains a file that produce a HTML report with trend projections on datas given by the data files. It utilizes sklearn's polynomial features, and functionTransformer to fit polynomial power of 1 through 5, and negative exponential line onto the data, and selects the one with the lowest MSE.

To get started:

- Add a folder "data", "data/trend_projection", "report", "plot", and "plot/trend_projection"

- Run "python - venv venv" in your terminal, make sure you are at the project root directory before running that code.

- Run "source venv/bin/activate" for mac users, "venv/Scripts/activate" for windows

- Then run "pip install -r requirements.txt"

- Then now you have to get your own data into the folder "data" for the hypothesis report, or "data/trend_projection" for trend projection.
    - You can utilize "pull_data.py" to pull data from influx database if you use influx database but you might have to modify the influx database field names or modify the code

- Then run "python report.py" when you have the 2 data files in csv to compare. Make sure to update the file path in the code in file "report.py"
    - Should produce 3 reports in folder 'report', HTML, JSON, and TXT files.