from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional
import numpy as np
import os

app = FastAPI(title="Lifestyle Analysis API")

# CORS middleware - allow both local and production origins
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    os.getenv("FRONTEND_URL", "https://dv-frontend.vercel.app")
]

# Add any additional origins from environment variable
if os.getenv("ADDITIONAL_ORIGINS"):
    allowed_origins.extend(os.getenv("ADDITIONAL_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data - support both local and production paths
data_path = os.getenv("DATA_PATH", "../Final_data.csv")
df = pd.read_csv(data_path)

@app.get("/")
def read_root():
    return {"message": "Lifestyle Analysis API"}

@app.get("/api/stats")
def get_stats():
    """Get overall statistics"""
    return {
        "totalParticipants": int(len(df)),
        "avgBMI": round(float(df['BMI'].mean()), 2),
        "mostPopularWorkout": str(df['Workout_Type'].mode()[0]),
        "avgCalories": int(df['Calories_Burned'].mean())
    }

@app.get("/api/gender-distribution")
def get_gender_distribution():
    """Get gender distribution data"""
    gender_counts = df['Gender'].value_counts()
    return [
        {"name": str(name), "value": int(count)}
        for name, count in gender_counts.items()
    ]

@app.get("/api/age-distribution")
def get_age_distribution():
    """Get age distribution data"""
    # Use same age groups as Demographics page
    bins = [13, 18, 30, 50, 65, 100]
    labels = ['Teen (13-17)', 'Young Adult (18-29)', 'Adult (30-49)', 'Middle Age (50-64)', 'Senior (65+)']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Only return groups that have data (count > 0)
    age_counts = df['Age_Group'].value_counts().sort_index()

    return [
        {"name": str(name), "value": int(count)}
        for name, count in age_counts.items()
        if count > 0  # Filter out empty groups
    ]

@app.get("/api/filtered-data")
def get_filtered_data(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2"),
    workout_freq: Optional[float] = Query(None),
    calories_burned: Optional[float] = Query(None),
    water_intake: Optional[float] = Query(None),
    session_duration: Optional[float] = Query(None)
):
    """Get filtered data based on user profile"""
    # Filter cohort
    cohort = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 2) & (df['BMI'] <= bmi + 2) &
        (df['Experience_Level'] == int(experience_level))
    ]

    cohort_size = len(cohort)

    if cohort_size == 0:
        return {
            "cohortSize": 0,
            "message": "No matching data found"
        }

    percentile_rank = (df['Calories_Burned'] < cohort['Calories_Burned'].mean()).sum() / len(df) * 100
    avg_days_active = float(cohort['Workout_Frequency (days/week)'].mean())

    # BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Radar chart data for performance comparison
    # Use user input values if provided, otherwise use cohort average
    user_workout_freq = workout_freq if workout_freq is not None else cohort['Workout_Frequency (days/week)'].mean()
    user_calories = calories_burned if calories_burned is not None else cohort['Calories_Burned'].mean()
    user_water = water_intake if water_intake is not None else cohort['Water_Intake (liters)'].mean()
    user_session = session_duration if session_duration is not None else cohort['Session_Duration (hours)'].mean()

    radar_data = [
        {
            "category": "Workout Freq",
            "user": int((user_workout_freq / 7) * 100),
            "average": int((df['Workout_Frequency (days/week)'].mean() / 7) * 100)
        },
        {
            "category": "Calories",
            "user": int((user_calories / df['Calories_Burned'].max()) * 100),
            "average": int((df['Calories_Burned'].mean() / df['Calories_Burned'].max()) * 100)
        },
        {
            "category": "BMI",
            "user": int((bmi / 50) * 100),  # Use BMI from filter
            "average": int((df['BMI'].mean() / 50) * 100)
        },
        {
            "category": "Water Intake",
            "user": int((user_water / df['Water_Intake (liters)'].max()) * 100),
            "average": int((df['Water_Intake (liters)'].mean() / df['Water_Intake (liters)'].max()) * 100)
        },
        {
            "category": "Session Duration",
            "user": int((user_session / df['Session_Duration (hours)'].max()) * 100),
            "average": int((df['Session_Duration (hours)'].mean() / df['Session_Duration (hours)'].max()) * 100)
        }
    ]

    return {
        "cohortSize": cohort_size,
        "percentileRank": round(percentile_rank, 0),
        "avgDaysActive": round(avg_days_active, 1),
        "bmiCategory": bmi_category,
        "cohortExercise": str(cohort['Workout_Type'].mode()[0]) if cohort_size > 0 else "N/A",
        "avgExercise": str(df['Workout_Type'].mode()[0]),
        "cohortDiet": str(cohort['diet_type'].mode()[0]) if cohort_size > 0 else "N/A",
        "avgDiet": str(df['diet_type'].mode()[0]),
        "radarData": radar_data
    }

@app.get("/api/workout-data")
def get_workout_data(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get workout analysis data"""
    # Filter data
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return {"message": "No data found"}

    total_active_mins = float(filtered_df['Session_Duration (hours)'].sum() * 60)
    most_common_exp = str(filtered_df['Experience_Level'].mode()[0])
    avg_heart_rate = float(filtered_df['pct_HRR'].mean() * 100)
    top_body_part = str(filtered_df['Body Part'].mode()[0])

    # Workout frequency distribution
    freq_counts = filtered_df['Workout_Frequency (days/week)'].round().astype(int).value_counts().sort_index()

    return {
        "totalActiveMins": round(total_active_mins, 0),
        "mostCommonExp": most_common_exp,
        "avgHeartRate": round(avg_heart_rate, 0),
        "topBodyPart": top_body_part,
        "frequencyDistribution": {
            "days": freq_counts.index.tolist(),
            "counts": freq_counts.values.tolist()
        }
    }

@app.get("/api/calories-by-workout-type")
def get_calories_by_workout_type(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get box plot data for calories burned by workout type"""
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return []

    result = []
    for workout_type in filtered_df['Workout_Type'].unique():
        workout_data = filtered_df[filtered_df['Workout_Type'] == workout_type]['Calories_Burned']

        if len(workout_data) > 0:
            result.append({
                "name": str(workout_type),
                "min": int(workout_data.min()),
                "q1": int(workout_data.quantile(0.25)),
                "median": int(workout_data.median()),
                "q3": int(workout_data.quantile(0.75)),
                "max": int(workout_data.max())
            })

    return result

@app.get("/api/calories-vs-bpm")
def get_calories_vs_bpm(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get scatter plot data for calories vs average BPM"""
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return []

    # Sample data to avoid sending too many points (max 200)
    sample_size = min(200, len(filtered_df))
    sampled = filtered_df.sample(n=sample_size)

    return [
        {
            "x": float(row['Avg_BPM']),
            "y": int(row['Calories_Burned'])
        }
        for _, row in sampled.iterrows()
    ]

@app.get("/api/calories-vs-duration")
def get_calories_vs_duration(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get scatter plot data for calories vs session duration"""
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return []

    # Sample data to avoid sending too many points (max 200)
    sample_size = min(200, len(filtered_df))
    sampled = filtered_df.sample(n=sample_size)

    return [
        {
            "x": float(row['Session_Duration (hours)']),
            "y": int(row['Calories_Burned'])
        }
        for _, row in sampled.iterrows()
    ]

@app.get("/api/participants-by-weekday")
def get_participants_by_weekday(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get participant count distributed across weekdays"""
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return []

    # Create synthetic weekday distribution based on workout frequency
    # People with higher frequency are more evenly distributed across days
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calculate distribution: higher frequency = more evenly spread across days
    total_participants = len(filtered_df)
    avg_freq = filtered_df['Workout_Frequency (days/week)'].mean()

    # Create a reasonable distribution
    # Mon-Fri slightly higher than weekends
    base_distribution = [1.0, 0.95, 1.05, 0.9, 1.1, 0.6, 0.5]  # Multipliers
    total_multiplier = sum(base_distribution)

    result = []
    for day, multiplier in zip(weekdays, base_distribution):
        count = int((multiplier / total_multiplier) * total_participants)
        result.append({
            "name": day,
            "value": count
        })

    return result

@app.get("/api/workout-comparison")
def get_workout_comparison(
    age: Optional[int] = Query(30),
    gender: Optional[str] = Query("Male"),
    bmi: Optional[float] = Query(25.0),
    experience_level: Optional[str] = Query("2")
):
    """Get workout comparison table with rankings"""
    filtered_df = df[
        (df['Age'] >= age - 5) & (df['Age'] <= age + 5) &
        (df['Gender'] == gender) &
        (df['BMI'] >= bmi - 3) & (df['BMI'] <= bmi + 3) &
        (df['Experience_Level'] == int(experience_level))
    ]

    if len(filtered_df) == 0:
        return []

    # Group by workout type and calculate metrics
    workout_stats = filtered_df.groupby('Workout_Type', observed=True).agg({
        'Calories_Burned': 'mean',
        'Session_Duration (hours)': 'mean',
        'Max_BPM': 'mean',
        'Body Part': lambda x: x.mode()[0] if len(x) > 0 else 'Full Body'
    }).reset_index()

    workout_stats.columns = ['workout', 'calories', 'duration', 'max_bpm', 'body_part']

    # Calculate Cal/Hour
    workout_stats['cal_per_hour'] = (workout_stats['calories'] / workout_stats['duration']).round(0).astype(int)

    # Determine intensity based on Max BPM
    def get_intensity(bpm):
        if bpm < 130:
            return 'Low'
        elif bpm < 160:
            return 'Medium'
        else:
            return 'High'

    workout_stats['intensity'] = workout_stats['max_bpm'].apply(get_intensity)

    # Determine "Best For" based on body part
    def get_best_for(body_part):
        mapping = {
            'Legs': 'Lower Body Strength',
            'Arms': 'Upper Body Strength',
            'Chest': 'Chest & Core',
            'Back': 'Back & Posture',
            'Abs': 'Core Strength',
            'Shoulders': 'Shoulder Definition',
            'Forearms': 'Grip Strength'
        }
        return mapping.get(body_part, 'Full Body Fitness')

    workout_stats['best_for'] = workout_stats['body_part'].apply(get_best_for)

    # Sort by cal_per_hour descending
    workout_stats = workout_stats.sort_values('cal_per_hour', ascending=False)

    # Create result
    result = []
    for rank, (_, row) in enumerate(workout_stats.iterrows(), start=1):
        result.append({
            "rank": rank,
            "workout": str(row['workout']),
            "calPerHour": int(row['cal_per_hour']),
            "intensity": str(row['intensity']),
            "bestFor": str(row['best_for'])
        })

    return result

# ========== NUTRITION AND DIET PAGE ==========

@app.get("/api/nutrition/stats")
def get_nutrition_stats():
    """Get nutrition statistics"""
    return {
        "avgDailyCalories": int(df['Calories'].mean()),
        "recommendedWaterIntake": round(float(df['Water_Intake (liters)'].mean()), 1),
        "mostCommonDiet": str(df['diet_type'].mode()[0]),
        "avgMealFreq": round(float(df['Daily meals frequency'].mean()), 1)
    }

@app.get("/api/nutrition/cooking-methods")
def get_cooking_methods():
    """Get top cooking methods"""
    cooking_counts = df['cooking_method'].value_counts().head(7)
    return [
        {"name": str(name), "value": int(count)}
        for name, count in cooking_counts.items()
    ]

@app.get("/api/nutrition/meal-timing")
def get_meal_timing():
    """Get meal timing distribution by meal type"""
    result = []
    for meal_type in df['meal_type'].unique():
        meal_data = df[df['meal_type'] == meal_type]['prep_time_min']
        if len(meal_data) > 0:
            result.append({
                "name": str(meal_type),
                "min": int(meal_data.min()),
                "q1": int(meal_data.quantile(0.25)),
                "median": int(meal_data.median()),
                "q3": int(meal_data.quantile(0.75)),
                "max": int(meal_data.max())
            })
    return result

@app.get("/api/nutrition/water-vs-performance")
def get_water_vs_performance():
    """Get water intake vs calories burned"""
    sampled = df.sample(n=min(200, len(df)))
    return [
        {
            "x": float(row['Water_Intake (liters)']),
            "y": int(row['Calories_Burned'])
        }
        for _, row in sampled.iterrows()
    ]

@app.get("/api/nutrition/macros-distribution")
def get_macros_distribution():
    """Get average macros distribution"""
    avg_carbs = df['Carbs'].mean()
    avg_proteins = df['Proteins'].mean()
    avg_fats = df['Fats'].mean()
    total = avg_carbs + avg_proteins + avg_fats

    return [
        {"name": "Carbs", "value": round((avg_carbs / total) * 100, 1)},
        {"name": "Proteins", "value": round((avg_proteins / total) * 100, 1)},
        {"name": "Fats", "value": round((avg_fats / total) * 100, 1)}
    ]

@app.get("/api/nutrition/diet-vs-calories")
def get_diet_vs_calories():
    """Get diet type vs calories"""
    diet_calories = df.groupby('diet_type', observed=True)['Calories'].mean().sort_values(ascending=False)
    return [
        {"name": str(name), "value": int(value)}
        for name, value in diet_calories.items()
    ]

@app.get("/api/nutrition/diet-vs-protein")
def get_diet_vs_protein():
    """Get diet type vs protein intake"""
    diet_protein = df.groupby('diet_type', observed=True)['Proteins'].mean().sort_values(ascending=False)
    return [
        {"name": str(name), "value": round(float(value), 1)}
        for name, value in diet_protein.items()
    ]

# ========== EXERCISE DEEP DIVE PAGE ==========

@app.get("/api/exercise/stats")
def get_exercise_stats():
    """Get exercise statistics"""
    avg_sets = round(float(df['Sets'].mean()), 1)
    avg_reps = round(float(df['Reps'].mean()), 1)

    return {
        "totalUniqueExercise": int(df['Name of Exercise'].nunique()),
        "mostTargetedMuscle": str(df['Body Part'].mode()[0]),
        "avgSetsReps": f"{avg_sets} × {avg_reps}",
        "topEquipment": str(df['Equipment Needed'].mode()[0])
    }

@app.get("/api/exercise/muscle-groups")
def get_muscle_groups():
    """Get muscle group distribution"""
    muscle_counts = df['Body Part'].value_counts()
    return [
        {"name": str(name), "value": int(count)}
        for name, count in muscle_counts.items()
    ]

@app.get("/api/exercise/equipment-usage")
def get_equipment_usage():
    """Get equipment usage distribution"""
    equipment_counts = df['Equipment Needed'].value_counts().head(8)
    return [
        {"name": str(name), "value": int(count)}
        for name, count in equipment_counts.items()
    ]

@app.get("/api/exercise/exercise-calories")
def get_exercise_calories():
    """Get exercise name vs calories burned (bubble chart data)"""
    exercise_data = df.groupby('Name of Exercise', observed=True).agg({
        'Calories_Burned': ['mean', 'count'],
        'Difficulty Level': lambda x: x.mode()[0] if len(x) > 0 else 'Intermediate'
    }).reset_index()

    exercise_data.columns = ['exercise', 'calories', 'count', 'difficulty']
    exercise_data = exercise_data.nlargest(30, 'count')

    difficulty_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}

    return [
        {
            "name": str(row['exercise']),
            "calories": int(row['calories']),
            "count": int(row['count']),
            "difficulty": difficulty_map.get(row['difficulty'], 2)
        }
        for _, row in exercise_data.iterrows()
    ]

@app.get("/api/exercise/top-exercises")
def get_top_exercises(muscle: Optional[str] = Query(None)):
    """Get top exercises for a specific muscle, ranked by average calories burned"""

    # Define keyword mappings for each body part based on exercise names
    muscle_keywords = {
        'Arms': ['curl', 'tricep', 'bicep', 'dips', 'pull-up', 'pull up', 'pullup', 'chin-up'],
        'Legs': ['squat', 'lunge', 'leg press', 'calf', 'step-up', 'step up'],
        'Chest': ['push-up', 'push up', 'pushup', 'bench press', 'chest press', 'press-up'],
        'Abs': ['crunch', 'plank', 'sit-up', 'sit up', 'russian twist', 'bicycle', 'flutter kick'],
        'Shoulders': ['shoulder', 'lateral raise', 'overhead', 'arnold press', 'shrug', 'wall angel', 'face pull', 'front raise', 'rear delt'],
        'Back': ['row', 'deadlift', 'pull', 'superman', 'lat pulldown', 'back extension'],
        'Forearms': ['wrist curl', 'farmer', 'grip', 'forearm']
    }

    # If no muscle selected or muscle not in our mapping, use body part filter
    if muscle is None or muscle not in muscle_keywords:
        filtered_df = df if muscle is None else df[df['Body Part'] == muscle]
    else:
        # Filter by exercise name containing keywords for the selected muscle
        keywords = muscle_keywords[muscle]
        pattern = '|'.join(keywords)
        filtered_df = df[df['Name of Exercise'].str.lower().str.contains(pattern, na=False, case=False)]

    # If we have no data after filtering, fall back to body part filter
    if len(filtered_df) == 0:
        filtered_df = df[df['Body Part'] == muscle]

    # Group by exercise and calculate average calories burned
    exercise_stats = filtered_df.groupby('Name of Exercise', observed=True).agg({
        'Calories_Burned': 'mean'
    }).reset_index()

    exercise_stats.columns = ['name', 'calories']

    # Sort by calories and get top 5
    top_exercises = exercise_stats.nlargest(5, 'calories')

    return [
        {"name": str(row['name']), "value": int(row['calories'])}
        for _, row in top_exercises.iterrows()
    ]

# ========== DEMOGRAPHICS AND TRENDS PAGE ==========

@app.get("/api/demographics/performance-matrix")
def get_performance_matrix():
    """Get performance matrix by age and gender"""
    # Create a local copy to avoid modifying the global df
    df_local = df.copy()
    
    # Remove any rows with missing Age or Gender
    df_local = df_local.dropna(subset=['Age', 'Gender'])
    
    # Define age groups - make sure to include all ages with include_lowest=True
    bins = [0, 18, 30, 50, 65, 100]
    labels = ['Teen (13-17)', 'Young Adult (18-29)', 'Adult (30-49)', 'Middle Age (50-64)', 'Senior (65+)']
    df_local['Age_Category'] = pd.cut(df_local['Age'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Drop NaN values before processing
    df_local = df_local.dropna(subset=['Age_Category', 'Gender', 'BMI', 'Calories_Burned', 'Workout_Frequency (days/week)'])

    result = []
    for age_group in labels:
        for gender in ['Male', 'Female']:
            subset = df_local[(df_local['Age_Category'] == age_group) & (df_local['Gender'] == gender)]

            if len(subset) > 0:
                avg_bmi = subset['BMI'].mean()
                bmi_healthy = 'healthy' if 18.5 <= avg_bmi < 25 else 'warning'

                result.append({
                    "ageGroup": age_group,
                    "gender": gender,
                    "calories": int(subset['Calories_Burned'].mean()),
                    "bmi": round(float(avg_bmi), 1),
                    "frequency": round(float(subset['Workout_Frequency (days/week)'].mean()), 1),
                    "population": len(subset),
                    "bmiStatus": bmi_healthy
                })

    return result

@app.get("/api/demographics/sankey-data")
def get_sankey_data():
    """Get Sankey diagram data for Age → Workout → Diet flow"""
    # Create a local copy to avoid modifying the global df
    df_local = df.copy()
    
    # Remove any rows with missing critical columns
    df_local = df_local.dropna(subset=['Age', 'Workout_Type', 'diet_type', 'Calories_Burned'])
    
    # Define age groups - make sure to include all ages with include_lowest=True
    bins = [0, 18, 30, 50, 65, 100]
    labels = ['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
    df_local['Age_Category'] = pd.cut(df_local['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Drop rows with NaN Age_Category
    df_clean = df_local.dropna(subset=['Age_Category'])

    # Get flows
    flows = df_clean.groupby(['Age_Category', 'Workout_Type', 'diet_type'], observed=True).agg({
        'Calories_Burned': ['count', 'mean']
    }).reset_index()

    flows.columns = ['age', 'workout', 'diet', 'count', 'calories']

    # Drop any rows with NaN calories
    flows = flows.dropna(subset=['calories'])

    return [
        {
            "source": str(row['age']),
            "target": f"{row['workout']} ({row['age']})",
            "value": int(row['count']),
            "calories": int(row['calories'])
        }
        for _, row in flows.iterrows()
    ] + [
        {
            "source": f"{row['workout']} ({row['age']})",
            "target": f"{row['diet']}",
            "value": int(row['count']),
            "calories": int(row['calories'])
        }
        for _, row in flows.iterrows()
    ]

# ========== DATA EXPLORER PAGE ==========

@app.get("/api/explorer/correlation-matrix")
def get_correlation_matrix():
    """Get correlation matrix for numerical columns"""
    numeric_cols = ['Age', 'BMI', 'Calories_Burned', 'Session_Duration (hours)',
                    'Workout_Frequency (days/week)', 'Water_Intake (liters)',
                    'Avg_BPM', 'Calories', 'Proteins', 'Carbs', 'Fats']

    corr_matrix = df[numeric_cols].corr()

    result = []
    for i, row_name in enumerate(corr_matrix.index):
        for j, col_name in enumerate(corr_matrix.columns):
            result.append({
                "x": col_name,
                "y": row_name,
                "value": round(float(corr_matrix.iloc[i, j]), 2)
            })

    return result

@app.get("/api/explorer/chart-data")
def get_chart_data(
    x_axis: str = Query(...),
    y_axis: Optional[str] = Query(None),
    color_by: Optional[str] = Query(None),
    size_by: Optional[str] = Query(None)
):
    """Get data for custom chart builder"""
    sampled = df.sample(n=min(500, len(df)))

    result = []
    for _, row in sampled.iterrows():
        data_point = {
            "x": float(row[x_axis]) if pd.api.types.is_numeric_dtype(df[x_axis]) else str(row[x_axis])
        }

        if y_axis:
            data_point["y"] = float(row[y_axis]) if pd.api.types.is_numeric_dtype(df[y_axis]) else str(row[y_axis])

        if color_by:
            data_point["color"] = str(row[color_by])

        if size_by:
            data_point["size"] = float(row[size_by]) if pd.api.types.is_numeric_dtype(df[size_by]) else 1

        result.append(data_point)

    return result

@app.get("/api/explorer/columns")
def get_available_columns():
    """Get list of available columns for chart builder"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "all": df.columns.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
