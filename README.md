# Traffic Accident Data Analysis

## Project Overview

This project performs an exploratory data analysis (EDA) of a large-scale US traffic accident dataset to identify patterns related to accident severity, weather conditions, visibility, road infrastructure, geographic hotspots, and temporal trends.

The analysis combines data preprocessing, feature engineering, statistical exploration, visualization, and geospatial mapping to uncover factors associated with traffic accidents and their severity.

---

## Objectives

The primary objectives of this project are:

- Analyze accident severity distribution.
- Identify accident trends by hour, day, and month.
- Examine the impact of weather conditions on accidents.
- Investigate the relationship between visibility and accident severity.
- Analyze road infrastructure features associated with accidents.
- Identify accident hotspot locations using geospatial visualization.
- Generate actionable insights through statistical summaries and visualizations.

---

## Dataset Information

### Dataset
US Traffic Accident Dataset (Sample)

### Dataset Size
- Approximately 100,000 records
- 46 columns

### Key Features Used

- Severity
- Start_Time
- State
- City
- Weather_Condition
- Temperature(F)
- Humidity(%)
- Visibility(mi)
- Pressure(in)
- Distance(mi)
- Sunrise_Sunset
- Start_Lat
- Start_Lng
- Junction
- Crossing
- Traffic_Signal
- Stop
- Amenity
- Bump

---

## Technologies Used

### Programming Language
- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Folium

---

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas and inspected for structure and quality.

### 2. Data Cleaning

The following preprocessing steps are performed:

- Converted `Start_Time` to datetime format.
- Removed rows containing invalid timestamps.
- Identified missing values.
- Dropped columns with more than 80% missing data.

### 3. Feature Engineering

Additional features were created to improve analysis:

- Hour
- Day
- Month
- DayOfWeek
- IsWeekend
- TimeOfDay

TimeOfDay categories:

- Night
- Morning
- Afternoon
- Evening
- Late Night

### 4. Exploratory Data Analysis

The project explores:

- Severity distribution
- State-wise accident distribution
- Weather condition frequency
- Hourly accident trends
- Daily and monthly accident patterns
- Road feature prevalence
- Visibility and severity relationships
- Temperature and severity relationships

### 5. Geospatial Analysis

Folium is used to generate:

- Accident hotspot heatmap
- Accident location map

These interactive maps allow geographic exploration of accident concentration areas.

---

## Visualizations Generated

### Figure 1: Overview Dashboard

Includes:

- Accident Severity Distribution
- Top 10 States by Accident Count
- Top 10 Weather Conditions
- Accidents by Hour of Day

Output:

```text
outputs/fig1_overview.png
```

---

### Figure 2: Time and Road Features

Includes:

- Accidents by Day of Week
- Accidents by Month
- Day vs Night Accident Distribution
- Road Feature Presence at Accident Sites

Output:

```text
outputs/fig2_time_road.png
```

---

### Figure 3: Deep Dive Analysis

Includes:

- Top 10 Cities by Accident Count
- Severity by Time of Day
- Temperature Distribution by Severity
- Correlation Heatmap

Output:

```text
outputs/fig3_deepdive.png
```

---

### Weather Condition vs Severity

Shows severity distribution under different weather conditions.

Output:

```text
outputs/weather_vs_severity.png
```

---

### Visibility vs Severity

Boxplot visualization showing how visibility levels relate to accident severity.

Output:

```text
outputs/visibility_vs_severity.png
```

---

## Interactive Maps

### Accident Hotspot Heatmap

Visualizes accident concentration using Folium HeatMap.

Output:

```text
outputs/Accident_Hotspots.html
```

---

### Accident Location Map

Displays sampled accident locations across the United States.

Output:

```text
outputs/Accident_Locations.html
```

---

## Key Findings

### Severity Distribution

Most accidents belong to Severity Levels 2 and 3, indicating moderate to serious incidents dominate the dataset.

### Geographic Concentration

Certain states and cities account for a disproportionately large number of accidents.

### Weather Impact

Weather conditions significantly influence accident occurrence patterns.

### Visibility Influence

Lower visibility conditions tend to be associated with more severe accidents.

### Road Infrastructure

Road elements such as:

- Junctions
- Traffic Signals
- Crossings

appear frequently in accident-prone locations.

### Time-Based Trends

Accident frequency increases during peak traffic periods and varies across weekdays and months.

### Hotspot Identification

Geospatial analysis highlights major accident concentration zones and high-risk regions.

---

## Project Structure

```text
Traffic_Accident_Data_Analysis/  
│
├── outputs/
│   ├── fig1_overview.png
│   ├── fig2_time_road.png
│   ├── fig3_deepdive.png
│   ├── weather_vs_severity.png
│   ├── visibility_vs_severity.png
│   ├── Accident_Hotspots.html
│   └── Accident_Locations.html
│
├── traffic_accident_data_analysis.py
├── Traffic_Accident_Data_Analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
```

### Navigate to Project Folder

```bash
cd Traffic_Accident_Data_Analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the Python script:

```bash
python traffic_accident_data_analysis.py
```

All generated outputs will automatically be saved inside the `outputs` folder.

---

## Requirements

Required Python packages:

```text
pandas
numpy
matplotlib
seaborn
folium
```

---

## Future Improvements

Potential enhancements include:

- Machine Learning models for accident severity prediction
- Real-time traffic accident analysis
- Interactive dashboard development using Streamlit or Dash
- Advanced geospatial clustering techniques
- Predictive risk assessment models

---

## Conclusion

This project demonstrates a complete data analysis workflow involving data cleaning, feature engineering, exploratory data analysis, statistical interpretation, visualization, and geospatial mapping.

The findings provide valuable insights into accident patterns related to weather conditions, visibility, traffic timing, road infrastructure, and geographic concentration, helping better understand factors associated with traffic accidents.

---

## Dataset

This project uses a sample of the US Accidents dataset for traffic accident analysis and visualization.

The dataset file is not included in this repository due to GitHub file size limitations.

To run the project locally, place the dataset file in the project directory and ensure the filename matches:

`accidents_sample.csv`

---

## Author

**Sriya Patil**

Aspiring Data Analyst and Machine Learning Enthusiast with a strong interest in data analysis, visualization, and machine learning.

- GitHub: https://github.com/SriyaPatil
- LinkedIn: https://linkedin.com/in/sriya-patil-63240332a

---


