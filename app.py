import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import defaultdict
import tempfile
import os
import json
from datetime import datetime, timedelta
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import re
import pytesseract
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Fitness Pro - Complete Health & Fitness Suite",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #4682B4;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .safety-safe { 
        background-color: #d4edda;
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #28a745; 
        margin: 1rem 0;
    }
    .safety-caution { 
        background-color: #fff3cd;   
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #ffc107; 
        margin: 1rem 0;
    }
    .safety-risky { 
        background-color: #f8d7da; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #dc3545; 
        margin: 1rem 0;
    }
    .safety-dangerous { 
        background-color: #f5c6cb; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #721c24; 
        margin: 1rem 0;
    }
    .nav-button {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .nav-button:hover {
        background-color: #e9ecef;
        cursor: pointer;
    }
    .progress-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin-bottom: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background-color: #28a745;
        border-radius: 5px;
        text-align: center;
        color: white;
        line-height: 20px;
        font-weight: bold;
    }
    .achievement-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        background-color: #ffc107;
        color: #212529;
        margin: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------------------------
# Workout Tracker Components
# ---------------------------
# Define muscle groups and exercises
muscle_groups = {
    1: ("Chest", {
        1: ("Push-up", "detect_pushup"),
        2: ("Bench Press", "detect_bench_press"),
        3: ("Chest Fly", "detect_chest_fly")
    }),
    2: ("Back", {
        1: ("Pull-up", "detect_pullup"),
        2: ("Bent-over Row", "detect_bent_over_row"),
        3: ("Lat Pulldown", "detect_lat_pulldown")
    }),
    3: ("Legs", {
        1: ("Squat", "detect_squat"),
        2: ("Lunge", "detect_lunge"),
        3: ("Leg Press", "detect_leg_press")
    }),
    4: ("Arms", {
        1: ("Bicep Curl", "detect_bicep_curl"),
        2: ("Tricep Extension", "detect_tricep_extension"),
        3: ("Hammer Curl", "detect_hammer_curl")
    }),
    5: ("Shoulders", {
        1: ("Shoulder Press", "detect_shoulder_press"),
        2: ("Lateral Raise", "detect_lateral_raise"),
        3: ("Front Raise", "detect_front_raise")
    }),
    6: ("Core", {
        1: ("Sit-up", "detect_situp"),
        2: ("Plank", "detect_plank"),
        3: ("Russian Twist", "detect_russian_twist")
    }),
    7: ("Miscellaneous", {
        1: ("Jumping Jack", "MISC"),
        2: ("Burpee", "MISC"),
        3: ("Mountain Climber", "MISC")
    })
}

# Calories burned per rep for different exercises (approximate)
calories_per_rep = {
    "Push-up": 0.32, "Bench Press": 0.5, "Chest Fly": 0.35,
    "Pull-up": 0.5, "Bent-over Row": 0.45, "Lat Pulldown": 0.4,
    "Squat": 0.6, "Lunge": 0.45, "Leg Press": 0.55,
    "Bicep Curl": 0.3, "Tricep Extension": 0.25, "Hammer Curl": 0.3,
    "Shoulder Press": 0.4, "Lateral Raise": 0.3, "Front Raise": 0.3,
    "Sit-up": 0.3, "Plank": 0.2, "Russian Twist": 0.25,
    "Jumping Jack": 0.2, "Burpee": 0.8, "Mountain Climber": 0.3
}

# Exercise difficulty levels (for achievements)
exercise_difficulty = {
    "Push-up": 3, "Bench Press": 4, "Chest Fly": 3,
    "Pull-up": 5, "Bent-over Row": 4, "Lat Pulldown": 3,
    "Squat": 4, "Lunge": 3, "Leg Press": 3,
    "Bicep Curl": 2, "Tricep Extension": 2, "Hammer Curl": 2,
    "Shoulder Press": 3, "Lateral Raise": 2, "Front Raise": 2,
    "Sit-up": 2, "Plank": 3, "Russian Twist": 2,
    "Jumping Jack": 1, "Burpee": 5, "Mountain Climber": 3
}

# Exercise detection functions (same as before)
def detect_pushup(landmarks, stage, counter):
    # Get coordinates
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    
    # Calculate angle
    angle = calculate_angle([left_shoulder.x, left_shoulder.y],
                           [left_elbow.x, left_elbow.y],
                           [left_wrist.x, left_wrist.y])
    
    # Push-up counter logic
    feedback = ""
    done = False
    
    if angle > 160:
        stage = "down"
    if angle < 90 and stage == "down":
        stage = "up"
        counter += 1
        done = True
        feedback = "Good form!"
    
    return stage, counter, angle, feedback, done

# Other detection functions remain the same as in your original code...

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def log_session(summary):
    # Create CSV file if it doesn't exist
    file_exists = os.path.isfile("workout_log.csv")
    
    with open("workout_log.csv", "a", newline="") as f:
        fieldnames = ["timestamp", "group", "exercise", "sets_completed", 
                     "target_sets", "total_reps", "calories", "difficulty"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "group": summary["group"],
            "exercise": summary["exercise"],
            "sets_completed": summary["sets_completed"],
            "target_sets": summary["target_sets"],
            "total_reps": summary["total_reps"],
            "calories": summary["calories"],
            "difficulty": exercise_difficulty.get(summary["exercise"], 3)
        })

# Placeholder functions for other exercises (same as before)
def detect_bench_press(landmarks, stage, counter):
    return stage, counter, 0, "", False

# Other placeholder functions remain the same...

# ---------------------------
# Nutrition Analyzer Components
# ---------------------------
def run_nutrition_ocr_with_health_analysis(image, selected_conditions):
    # Placeholder implementation for OCR and health analysis
    # In a real implementation, this would use Tesseract or another OCR engine
    # and analyze the extracted nutrition data against health conditions
    
    # For demo purposes, we'll return mock data
    nutrition_data = {
        "nutrition_facts": {
            "Calories": "250 kcal",
            "Protein": "15g",
            "Carbohydrates": "30g",
            "Sugar": "12g",
            "Fat": "8g",
            "Saturated Fat": "3g",
            "Sodium": "450mg"
        },
        "ingredients": ["Wheat flour", "Sugar", "Palm oil", "Salt", "Natural flavors"]
    }
    
    health_analysis = {
        "overall_safety": "CAUTION",
        "risk_level": "Moderate",
        "problematic_ingredients": [
            {
                "ingredient": "Sugar",
                "reason": "High sugar content may impact blood glucose levels",
                "severity": "Moderate"
            },
            {
                "ingredient": "Palm oil",
                "reason": "Contains saturated fats that may affect cholesterol",
                "severity": "Low"
            }
        ],
        "risky_nutrients": [
            {
                "nutrient": "Sodium",
                "current_value": "450mg",
                "max_recommended": "2300mg",
                "reason": "Moderate sodium content, should be consumed in moderation for hypertension"
            }
        ],
        "recommendations": [
            "Consume in moderation due to sugar content",
            "Pair with protein-rich foods to balance blood sugar response",
            "Monitor portion size if you have diabetes or heart conditions"
        ]
    }
    
    return [("Tesseract", nutrition_data, "Mock extracted text")], ("Tesseract", nutrition_data, "Mock extracted text"), health_analysis

def display_analysis_results(health_analysis, nutrition_data):
    # Display nutrition facts
    st.subheader("📊 Nutrition Facts")
    for key, value in nutrition_data.get("nutrition_facts", {}).items():
        st.write(f"**{key}:** {value}")
    
    # Display ingredients
    if nutrition_data.get("ingredients"):
        st.subheader("🧪 Ingredients")
        for i, ingredient in enumerate(nutrition_data["ingredients"], 1):
            st.write(f"{i}. {ingredient}")
    
    # Display health analysis
    st.subheader("🏥 Health Analysis")
    
    # Safety overview
    safety_level = health_analysis.get("overall_safety", "UNKNOWN")
    safety_classes = {
        "SAFE": "safety-safe",
        "CAUTION": "safety-caution", 
        "RISKY": "safety-risky",
        "DANGEROUS": "safety-dangerous"
    }
    
    st.markdown(f'<div class="{safety_classes.get(safety_level, "safety-caution")}">'
                f'<h3>Overall Safety: {safety_level}</h3>'
                f'<p>Risk Level: {health_analysis.get("risk_level", "Unknown")}</p>'
                f'</div>', unsafe_allow_html=True)
    
    # Problematic ingredients
    if health_analysis.get("problematic_ingredients"):
        st.subheader("⚠️ Problematic Ingredients")
        for issue in health_analysis["problematic_ingredients"]:
            st.warning(f"**{issue['ingredient']}** - {issue['reason']} ({issue['severity']} severity)")
    
    # Risky nutrients
    if health_analysis.get("risky_nutrients"):
        st.subheader("📈 Nutrient Concerns")
        for nutrient in health_analysis["risky_nutrients"]:
            st.info(f"**{nutrient['nutrient']}**: {nutrient['current_value']} "
                   f"(Recommended: < {nutrient['max_recommended']}) - {nutrient['reason']}")
    
    # Recommendations
    if health_analysis.get("recommendations"):
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(health_analysis["recommendations"], 1):
            st.write(f"{i}. {rec}")

# ---------------------------
# Diet Plan Generator Components
# ---------------------------
class DietChart(TypedDict):
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    goal: str
    preferences: dict
    medical_conditions: List[str]
    bmr: float
    tdee: float
    macros: dict
    meal_timing: str
    diet_chart: str
    grocery_list: str
    progress_tracking: str

def collect_user_requirements():
    """Collect comprehensive user information for diet planning"""
    st.subheader("Personal Information")
    age = st.number_input("Enter your age", min_value=15, max_value=100, value=30)
    gender = st.selectbox("Enter your gender", ["male", "female"])
    weight = st.number_input("Enter your weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Enter your height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    
    st.subheader("Activity & Goals")
    activity_level = st.selectbox("Enter activity level", 
                                 ["sedentary", "light", "moderate", "active", "very active"])
    goal = st.selectbox("Enter your goal", ["weight loss", "muscle gain", "maintenance"])
    
    st.subheader("Food Preferences")
    diet_type = st.selectbox("Enter diet type", 
                            ["vegetarian", "non-vegetarian", "vegan", "gluten-free"])
    allergies = st.text_input("Enter allergies (comma separated, or leave blank if none)")
    dislikes = st.text_input("Enter foods you dislike (comma separated, or leave blank)")
    
    st.subheader("Lifestyle & Health")
    medical_conditions = st.text_input("Enter any medical conditions (comma separated, or leave blank)")
    meal_timing = st.selectbox("Preferred meal timing", ["early eater", "late eater", "flexible"])
    cooking_ability = st.selectbox("Cooking ability", ["beginner", "intermediate", "advanced"])
    
    allergies_list = [a.strip().lower() for a in allergies.split(",")] if allergies else []
    dislikes_list = [d.strip().lower() for d in dislikes.split(",")] if dislikes else []
    medical_list = [m.strip().lower() for m in medical_conditions.split(",")] if medical_conditions else []
    
    return {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "goal": goal,
        "preferences": {
            "diet": diet_type,
            "allergies": allergies_list,
            "dislikes": dislikes_list,
            "cooking_ability": cooking_ability
        },
        "medical_conditions": medical_list,
        "meal_timing": meal_timing
    }

def validate_inputs(state: DietChart) -> DietChart:
    """Validate user inputs with comprehensive checks"""
    required = ["age", "gender", "weight", "height", "activity_level", "goal"]
    for r in required:
        if r not in state or state[r] is None:
            raise ValueError(f"Missing input: {r}")
    
    # Validate specific ranges
    if state["age"] < 15 or state["age"] > 100:
        raise ValueError("Age must be between 15 and 100")
    
    if state["weight"] < 30 or state["weight"] > 200:
        raise ValueError("Weight must be between 30 and 200 kg")
    
    if state["height"] < 100 or state["height"] > 250:
        raise ValueError("Height must be between 100 and 250 cm")
    
    valid_genders = ["male", "female"]
    if state["gender"] not in valid_genders:
        raise ValueError("Gender must be 'male' or 'female'")
    
    valid_activity = ["sedentary", "light", "moderate", "active", "very active"]
    if state["activity_level"] not in valid_activity:
        raise ValueError("Invalid activity level")
    
    valid_goals = ["weight loss", "muscle gain", "maintenance"]
    if state["goal"] not in valid_goals:
        raise ValueError("Invalid goal")
    
    return state

def calculate_macros(state: DietChart) -> DietChart:
    """Calculate BMR, TDEE, and macros with goal-specific adjustments"""
    # Mifflin-St Jeor Equation for BMR
    if state["gender"] == "male":
        bmr = 10 * state["weight"] + 6.25 * state["height"] - 5 * state["age"] + 5
    else:
        bmr = 10 * state["weight"] + 6.25 * state["height"] - 5 * state["age"] - 161
    
    activity_factor = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }
    tdee = bmr * activity_factor.get(state["activity_level"], 1.55)

    # Adjust based on goal
    goal_adjustments = {
        "weight loss": 0.85,  # 15% deficit
        "maintenance": 1.0,   # maintain
        "muscle gain": 1.15   # 15% surplus
    }
    tdee = tdee * goal_adjustments.get(state["goal"], 1.0)

    # Macro split based on goal
    if state["goal"] == "weight loss":
        macros = {
            "protein": (0.35 * tdee) / 4,  # 35% protein
            "carbs": (0.40 * tdee) / 4,    # 40% carbs
            "fat": (0.25 * tdee) / 9       # 25% fat
        }
    elif state["goal"] == "muscle gain":
        macros = {
            "protein": (0.30 * tdee) / 4,  # 30% protein
            "carbs": (0.50 * tdee) / 4,    # 50% carbs
            "fat": (0.20 * tdee) / 9       # 20% fat
        }
    else:  # maintenance
        macros = {
            "protein": (0.25 * tdee) / 4,  # 25% protein
            "carbs": (0.45 * tdee) / 4,    # 45% carbs
            "fat": (0.30 * tdee) / 9       # 30% fat
        }

    return {"bmr": round(bmr, 2), "tdee": round(tdee, 2), "macros": macros}

def generate_diet_chart(state: DietChart) -> DietChart:
    """Generate a personalized diet chart using the LLM"""
    # Check if API key is available
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("Please set the GOOGLE_API_KEY environment variable to use the diet plan generator")
        return {
            "diet_chart": "API key not available. Please set GOOGLE_API_KEY environment variable.",
            "grocery_list": "API key not available.",
            "progress_tracking": "API key not available."
        }
    
    # Initialize the Gemini model
    model = ChatGoogleGenerativeAI(
        model='models/gemini-2.5-flash',
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.7
    )
    
    prompt = f"""
    Create a personalized diet plan based on the following information:
    
    Personal Details:
    - Age: {state['age']}
    - Gender: {state['gender']}
    - Weight: {state['weight']} kg
    - Height: {state['height']} cm
    
    Goals & Activity:
    - Activity Level: {state['activity_level']}
    - Goal: {state['goal']}
    
    Preferences & Restrictions:
    - Diet Type: {state['preferences']['diet']}
    - Allergies: {', '.join(state['preferences']['allergies']) if state['preferences']['allergies'] else 'None'}
    - Dislikes: {', '.join(state['preferences']['dislikes']) if state['preferences']['dislikes'] else 'None'}
    - Medical Conditions: {', '.join(state['medical_conditions']) if state['medical_conditions'] else 'None'}
    - Meal Timing Preference: {state['meal_timing']}
    - Cooking Ability: {state['preferences']['cooking_ability']}
    
    Nutritional Targets:
    - Daily Calories: {state['tdee']} kcal
    - Protein: {state['macros']['protein']:.1f}g
    - Carbohydrates: {state['macros']['carbs']:.1f}g
    - Fat: {state['macros']['fat']:.1f}g
    
    Please create:
    1. A 7-day meal plan with breakfast, lunch, dinner, and 2 snacks
    2. Portion sizes for each meal
    3. Cooking instructions for complex dishes
    4. Alternatives for each meal for variety
    5. Hydration recommendations
    """
    
    diet_response = model.invoke(prompt)
    
    # Generate grocery list
    grocery_prompt = f"""
    Based on the diet plan you just created, generate a comprehensive grocery list 
    organized by categories (produce, proteins, grains, dairy, etc.) with quantities 
    for one week of meals. Consider that the person has: 
    - Diet type: {state['preferences']['diet']}
    - Allergies: {', '.join(state['preferences']['allergies'])}
    - Cooking ability: {state['preferences']['cooking_ability']}
    """
    
    grocery_response = model.invoke(grocery_prompt)
    
    # Generate progress tracking guidance
    progress_prompt = f"""
    Create a progress tracking guide for someone with these goals: {state['goal']}.
    Include:
    1. Key metrics to track (weight, measurements, etc.)
    2. Recommended frequency of tracking
    3. How to adjust the diet based on progress
    4. Non-scale victories to look for
    5. Timeline for expected results
    """
    
    progress_response = model.invoke(progress_prompt)
    
    return {
        "diet_chart": diet_response.content,
        "grocery_list": grocery_response.content,
        "progress_tracking": progress_response.content
    }

def save_results(state: DietChart):
    """Save the results to formatted files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diet_plan_{timestamp}"
    
    # Create JSON output
    json_output = {
        "personal_info": {
            "age": state["age"],
            "gender": state["gender"],
            "weight": state["weight"],
            "height": state["height"]
        },
        "goals": {
            "activity_level": state["activity_level"],
            "goal": state["goal"]
        },
        "preferences": state["preferences"],
        "medical_conditions": state["medical_conditions"],
        "nutritional_targets": {
            "bmr": state["bmr"],
            "tdee": state["tdee"],
            "macros": state["macros"]
        },
        "generated_on": datetime.now().isoformat()
    }
    
    # Save JSON file
    with open(f"{filename}.json", "w") as f:
        json.dump(json_output, f, indent=2)
    
    # Save detailed diet plan as text file
    with open(f"{filename}_diet_plan.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PERSONALIZED DIET PLAN\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERSONAL INFORMATION:\n")
        f.write(f"Age: {state['age']}\n")
        f.write(f"Gender: {state['gender']}\n")
        f.write(f"Weight: {state['weight']} kg\n")
        f.write(f"Height: {state['height']} cm\n\n")
        
        f.write("GOALS & ACTIVITY:\n")
        f.write(f"Activity Level: {state['activity_level']}\n")
        f.write(f"Goal: {state['goal']}\n\n")
        
        f.write("NUTRITIONAL TARGETS:\n")
        f.write(f"BMR: {state['bmr']} kcal\n")
        f.write(f"TDEE: {state['tdee']} kcal\n")
        f.write(f"Protein: {state['macros']['protein']:.1f}g\n")
        f.write(f"Carbohydrates: {state['macros']['carbs']:.1f}g\n")
        f.write(f"Fat: {state['macros']['fat']:.1f}g\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("7-DAY MEAL PLAN\n")
        f.write("=" * 60 + "\n\n")
        f.write(state["diet_chart"] + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("GROCERY LIST\n")
        f.write("=" * 60 + "\n\n")
        f.write(state["grocery_list"] + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("PROGRESS TRACKING GUIDE\n")
        f.write("=" * 60 + "\n\n")
        f.write(state["progress_tracking"] + "\n")
    
    st.success(f"Results saved to {filename}.json and {filename}_diet_plan.txt")
    return state

# ---------------------------
# Navigation
# ---------------------------
def navigation():
    st.sidebar.markdown("## 🧭 Navigation")
    
    # App sections
    sections = {
        "🏠 Dashboard": "dashboard",
        "💪 Workout Tracker": "workout",
        "🍎 Nutrition Analyzer": "nutrition",
        "📋 Diet Plan Generator": "diet",
        "📊 Progress & Analytics": "progress"
    }
    
    selected = st.sidebar.radio("Go to", list(sections.keys()))
    
    # Return the selected section key
    return sections[selected]

# ---------------------------
# Enhanced Dashboard
# ---------------------------
def show_dashboard():
    st.markdown('<h1 class="main-header">🏋️ Fitness Pro - Complete Health Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your all-in-one solution for fitness tracking, nutrition analysis, and diet planning</p>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics from workout history
    try:
        with open("workout_log.csv", "r") as f:
            reader = csv.reader(f)
            workouts = list(reader)
            if len(workouts) > 1:  # Skip header
                total_workouts = len(workouts) - 1
                total_reps = sum(int(row[5]) for row in workouts[1:])
                total_calories = sum(float(row[6]) for row in workouts[1:])
                last_workout = workouts[-1][2] if len(workouts) > 1 else "No workouts"
            else:
                total_workouts = 0
                total_reps = 0
                total_calories = 0
                last_workout = "No workouts"
    except FileNotFoundError:
        total_workouts = 0
        total_reps = 0
        total_calories = 0
        last_workout = "No workouts"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Workouts</h3>
            <p style="font-size: 2rem; font-weight: bold; text-align: center; margin: 0;">{total_workouts}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💪 Total Reps</h3>
            <p style="font-size: 2rem; font-weight: bold; text-align: center; margin: 0;">{total_reps}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Calories Burned</h3>
            <p style="font-size: 2rem; font-weight: bold; text-align: center; margin: 0;">{total_calories:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Last Workout</h3>
            <p style="font-size: 1.2rem; font-weight: bold; text-align: center; margin: 0;">{last_workout}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Weekly progress
    st.markdown("## 📈 Weekly Progress")
    
    try:
        # Read workout data
        df = pd.read_csv("workout_log.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        # Get current week data
        current_week = datetime.now().isocalendar().week
        week_data = df[df['week'] == current_week]
        
        # Calculate weekly metrics
        weekly_workouts = len(week_data)
        weekly_reps = week_data['total_reps'].sum()
        weekly_calories = week_data['calories'].sum()
        
        # Create columns for weekly metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Workouts This Week", weekly_workouts)
        
        with col2:
            st.metric("Reps This Week", weekly_reps)
        
        with col3:
            st.metric("Calories This Week", f"{weekly_calories:.0f}")
        
        # Weekly workout distribution
        if not week_data.empty:
            st.markdown("#### Workout Distribution This Week")
            fig = px.pie(week_data, names='exercise', title='Workouts by Type')
            st.plotly_chart(fig, use_container_width=True)
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.info("No workout data available. Start tracking to see your progress!")
    
    # Recent activity
    st.markdown("## 📋 Recent Activity")
    
    try:
        with open("workout_log.csv", "r") as f:
            reader = csv.reader(f)
            workouts = list(reader)
            if len(workouts) > 1:  # Skip header
                st.markdown("#### Recent Workouts")
                for i, workout in enumerate(workouts[-5:][::-1]):  # Show last 5
                    if i == 0: continue  # Skip header
                    st.write(f"**{workout[0]}** - {workout[2]} ({workout[5]} reps, {workout[6]} kcal)")
            else:
                st.info("No workout history yet. Start tracking your workouts!")
    except FileNotFoundError:
        st.info("No workout history yet. Start tracking your workouts!")
    
    # Quick actions
    st.markdown("## ⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💪 Start New Workout", use_container_width=True):
            st.session_state.current_section = "workout"
            st.rerun()
    
    with col2:
        if st.button("🍎 Analyze Nutrition", use_container_width=True):
            st.session_state.current_section = "nutrition"
            st.rerun()
    
    with col3:
        if st.button("📋 Generate Diet Plan", use_container_width=True):
            st.session_state.current_section = "diet"
            st.rerun()
    
    with col4:
        if st.button("📊 View Progress", use_container_width=True):
            st.session_state.current_section = "progress"
            st.rerun()
    
    # Motivational quote
    st.markdown("---")
    quotes = [
        "The only bad workout is the one that didn't happen.",
        "Your body can stand almost anything. It's your mind that you have to convince.",
        "The hardest lift of all is lifting your butt off the couch.",
        "Strength doesn't come from what you can do. It comes from overcoming the things you once thought you couldn't.",
        "Success starts with self-discipline."
    ]
    
    st.markdown(f'<div style="text-align: center; font-style: italic; margin: 20px 0;">"{np.random.choice(quotes)}"</div>', unsafe_allow_html=True)




def detector_misc(landmarks, misc_state):
    # Get coordinates
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    # For miscellaneous exercises like jumping jacks
    feedback = ""
    done = False
    
    if misc_state["prev_wrist_y"] is None:
        misc_state["prev_wrist_y"] = (left_wrist.y + right_wrist.y) / 2
    
    avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
    
    # Simple logic for jumping motion
    if abs(avg_wrist_y - misc_state["prev_wrist_y"]) > 0.05:
        misc_state["cooldown"] += 1
        if misc_state["cooldown"] > 5:  # Count after several frames of movement
            done = True
            misc_state["cooldown"] = 0
            feedback = "Good repetition!"
    
    misc_state["prev_wrist_y"] = avg_wrist_y
    
    return misc_state, done, feedback




def detect_pullup(landmarks, stage, counter, prev_shoulder_y=None):
    # Get coordinates
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Calculate shoulder height (average of both shoulders)
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    feedback = ""
    done = False
    
    if prev_shoulder_y is None:
        prev_shoulder_y = shoulder_y
    
    # Pull-up counter logic
    if shoulder_y < prev_shoulder_y - 0.05:  # Shoulders moving up
        stage = "up"
    elif shoulder_y > prev_shoulder_y + 0.05:  # Shoulders moving down
        if stage == "up":
            stage = "down"
            counter += 1
            done = True
            feedback = "Good pull-up!"
    
    return stage, counter, shoulder_y, feedback, done, shoulder_y




# ---------------------------
# Workout Tracker (unchanged)
# ---------------------------
# Workout Tracker
# ---------------------------
def workout_tracker():
    st.title("💪 Workout Tracker")
    
    # Initialize session state variables for workout
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'stage' not in st.session_state:
        st.session_state.stage = None
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'sets_completed' not in st.session_state:
        st.session_state.sets_completed = 0
    if 'total_reps_done' not in st.session_state:
        st.session_state.total_reps_done = 0
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""
    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = None
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
    if 'target_reps' not in st.session_state:
        st.session_state.target_reps = 10
    if 'target_sets' not in st.session_state:
        st.session_state.target_sets = 3
    if 'rest_between_sets' not in st.session_state:
        st.session_state.rest_between_sets = 30
    if 'prev_shoulder_y' not in st.session_state:
        st.session_state.prev_shoulder_y = None
    if 'misc_state' not in st.session_state:
        st.session_state.misc_state = {"prev_wrist_y": None, "cooldown": 0}
    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'pose' not in st.session_state:
        st.session_state.pose = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Workout Configuration")
        
        # Muscle group selection
        group_options = {k: v[0] for k, v in muscle_groups.items()}
        selected_group_num = st.selectbox(
            "Select Muscle Group:",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x]
        )
        st.session_state.selected_group = muscle_groups[selected_group_num]
        
        # Exercise selection
        if st.session_state.selected_group:
            exercise_options = {k: v[0] for k, v in st.session_state.selected_group[1].items()}
            selected_exercise_num = st.selectbox(
                "Select Exercise:",
                options=list(exercise_options.keys()),
                format_func=lambda x: exercise_options[x]
            )
            st.session_state.selected_exercise = st.session_state.selected_group[1][selected_exercise_num]
        
        # Target settings
        st.session_state.target_reps = st.number_input("Target reps per set:", min_value=1, value=10)
        st.session_state.target_sets = st.number_input("Target sets:", min_value=1, value=3)
        st.session_state.rest_between_sets = st.number_input("Rest between sets (seconds):", min_value=5, value=30)
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Workout") and not st.session_state.running:
                st.session_state.running = True
                st.session_state.stage = None
                st.session_state.counter = 0
                st.session_state.sets_completed = 0
                st.session_state.total_reps_done = 0
                st.session_state.feedback = ""
                st.session_state.prev_shoulder_y = None
                st.session_state.misc_state = {"prev_wrist_y": None, "cooldown": 0}
                st.session_state.show_summary = False
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                st.rerun()
        
        with col2:
            if st.button("Stop Workout") and st.session_state.running:
                st.session_state.running = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                if st.session_state.pose:
                    st.session_state.pose.close()
                st.rerun()
                
        # Show history
        if st.button("Show Workout History"):
            try:
                with open("workout_log.csv", "r") as f:
                    st.download_button(
                        "Download Workout History",
                        f.read(),
                        "workout_history.csv",
                        "text/csv"
                    )
            except FileNotFoundError:
                st.warning("No workout history found")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Camera Feed")
        video_placeholder = st.empty()
        
        if st.session_state.running and st.session_state.cap and st.session_state.pose:
            # Workout loop
            while st.session_state.running and st.session_state.sets_completed < st.session_state.target_sets:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Cannot access webcam")
                    st.session_state.running = False
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Process with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = st.session_state.pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Exercise detection
                if results.pose_landmarks and st.session_state.selected_exercise:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Check if it's a pull-up (special case)
                    is_pullup = st.session_state.selected_exercise[0] == "Pull-up"
                    
                    # Check if it's a miscellaneous exercise
                    is_misc = st.session_state.selected_exercise[1] == "MISC"
                    
                    if is_misc:
                        st.session_state.misc_state, done, st.session_state.feedback = detector_misc(
                            landmarks, st.session_state.misc_state
                        )
                        if done:
                            st.session_state.counter += 1
                            st.session_state.total_reps_done += 1
                    elif is_pullup:
                        st.session_state.stage, st.session_state.counter, _, st.session_state.feedback, done, st.session_state.prev_shoulder_y = detect_pullup(
                            landmarks, st.session_state.stage, st.session_state.counter, st.session_state.prev_shoulder_y
                        )
                        if done:
                            st.session_state.total_reps_done += 1
                    else:
                        detector_name = st.session_state.selected_exercise[1]
                        # Map detector name to function
                        detector_func = globals().get(detector_name, detect_pushup)
                        st.session_state.stage, st.session_state.counter, _, st.session_state.feedback, done = detector_func(
                            landmarks, st.session_state.stage, st.session_state.counter
                        )
                        if done:
                            st.session_state.total_reps_done += 1
                
                # Check if set is completed
                if st.session_state.counter >= st.session_state.target_reps:
                    st.session_state.sets_completed += 1
                    st.session_state.counter = 0
                    
                    # If more sets to do, show rest timer
                    if st.session_state.sets_completed < st.session_state.target_sets:
                        with col2:
                            rest_placeholder = st.empty()
                            for i in range(st.session_state.rest_between_sets, 0, -1):
                                rest_placeholder.info(f"Resting: {i} seconds remaining")
                                time.sleep(1)
                            rest_placeholder.empty()
                
                # Draw landmarks and info on frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                
                # Add workout info overlay
                cv2.rectangle(image, (0, 0), (640, 100), (36, 120, 200), -1)
                cv2.putText(image, f"Sets: {st.session_state.sets_completed}/{st.session_state.target_sets}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Reps: {st.session_state.counter}/{st.session_state.target_reps}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Feedback: {st.session_state.feedback}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                video_placeholder.image(image, channels="BGR", use_container_width=True)
                
                # Small delay
                time.sleep(0.01)
            
            # Workout completed
            if st.session_state.sets_completed >= st.session_state.target_sets:
                st.session_state.running = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                if st.session_state.pose:
                    st.session_state.pose.close()
                
                # Calculate calories
                exercise_name = st.session_state.selected_exercise[0]
                cals = round(st.session_state.total_reps_done * calories_per_rep[exercise_name], 2)
                
                # Session summary
                session_summary = {
                    "group": st.session_state.selected_group[0],
                    "exercise": exercise_name,
                    "sets_completed": st.session_state.sets_completed,
                    "target_sets": st.session_state.target_sets,
                    "total_reps": st.session_state.total_reps_done,
                    "calories": cals
                }
                
                # Log session
                log_session(session_summary)
                
                # Show summary
                st.session_state.show_summary = True
                st.rerun()
                
        elif not st.session_state.running:
            video_placeholder.info("Click 'Start Workout' to begin tracking")
    
    with col2:
        st.header("Workout Status")
        
        if st.session_state.selected_group and st.session_state.selected_exercise:
            st.subheader(st.session_state.selected_group[0])
            st.write(f"**Exercise:** {st.session_state.selected_exercise[0]}")
            
            # Progress indicators
            st.metric("Sets Completed", 
                     f"{st.session_state.sets_completed} / {st.session_state.target_sets}")
            st.metric("Current Reps", 
                     f"{st.session_state.counter} / {st.session_state.target_reps}")
            st.metric("Total Reps", st.session_state.total_reps_done)
            
            # Feedback
            if st.session_state.feedback:
                st.info(f"Feedback: {st.session_state.feedback}")
            
            # Show summary if workout completed
            if st.session_state.show_summary:
                st.balloons()
                st.success("Workout completed! 🎉")
                
                # Calculate calories
                exercise_name = st.session_state.selected_exercise[0]
                cals = round(st.session_state.total_reps_done * calories_per_rep[exercise_name], 2)
                
                st.subheader("Session Summary")
                st.write(f"**Group:** {st.session_state.selected_group[0]}")
                st.write(f"**Exercise:** {exercise_name}")
                st.write(f"**Sets completed:** {st.session_state.sets_completed}/{st.session_state.target_sets}")
                st.write(f"**Total reps:** {st.session_state.total_reps_done}")
                st.write(f"**Estimated calories burned:** {cals} kcal")
            
            # Instructions
            st.subheader("Instructions")
            st.write("1. Position yourself in front of the camera")
            st.write("2. Make sure your entire body is visible")
            st.write("3. Perform the exercise with proper form")
            st.write("4. The system will count your reps automatically")

# -
# 

def create_json_report(health_analysis, nutrition_data, method_name):
    """Create a downloadable JSON report"""
    report_data = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "best_ocr_method": method_name,
            "analysis_version": "1.0"
        },
        "health_analysis": health_analysis,
        "nutrition_data": nutrition_data
    }
    return json.dumps(report_data, indent=2)

# -


def create_text_report(health_analysis, nutrition_data, method_name, selected_conditions):
    """Create a downloadable text report"""
    report = []
    report.append("🏥 COMPLETE NUTRITION & HEALTH ANALYSIS")
    report.append("=" * 60)
    report.append("")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Best OCR Method: {method_name}")
    report.append(f"Health Conditions Analyzed: {', '.join([c.replace('_', ' ').title() for c in selected_conditions])}")
    report.append("")
    
    # Health Safety Overview
    if health_analysis:
        safety_emoji = {
            "SAFE": "✅", "CAUTION": "⚠️", "RISKY": "🚨", "DANGEROUS": "🆘"
        }
        
        report.append("🛡️ HEALTH SAFETY OVERVIEW:")
        report.append(f"  Overall Safety: {safety_emoji.get(health_analysis['overall_safety'], '❓')} {health_analysis['overall_safety']}")
        report.append(f"  Risk Level: {health_analysis['risk_level']}")
        report.append(f"  Total Warnings: {len(health_analysis.get('problematic_ingredients', [])) + len(health_analysis.get('risky_nutrients', []))}")
        report.append("")
        
        # Problematic Ingredients
        if health_analysis.get("problematic_ingredients"):
            report.append("🧪 PROBLEMATIC INGREDIENTS FOUND:")
            for issue in health_analysis["problematic_ingredients"]:
                report.append(f"  ⚠️ {issue['ingredient']}")
                report.append(f"     Reason: {issue['reason']}")
                report.append(f"     Severity: {issue['severity']}")
                report.append("")
        
        # Risky Nutrients
        if health_analysis.get("risky_nutrients"):
            report.append("📊 RISKY NUTRIENT LEVELS:")
            for nutrient in health_analysis["risky_nutrients"]:
                report.append(f"  🚨 {nutrient['nutrient']}")
                report.append(f"     Current: {nutrient['current_value']}")
                report.append(f"     Safe Limit: {nutrient['max_recommended']}")
                report.append(f"     Impact: {nutrient['reason']}")
                report.append("")
        
        # Recommendations
        if health_analysis.get("recommendations"):
            report.append("💡 PERSONALIZED RECOMMENDATIONS:")
            for i, rec in enumerate(health_analysis["recommendations"], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
    
    # Nutrition Facts
    if nutrition_data.get("nutrition_facts"):
        report.append("📊 NUTRITION FACTS:")
        for key, value in nutrition_data["nutrition_facts"].items():
            report.append(f"  • {key}: {value}")
        report.append("")
    
    # Ingredients
    if nutrition_data.get("ingredients"):
        report.append("🧪 INGREDIENTS:")
        for i, ingredient in enumerate(nutrition_data["ingredients"], 1):
            report.append(f"  {i}. {ingredient}")
        report.append("")
    
    report.append("⚠️ DISCLAIMER:")
    report.append("This analysis is for informational purposes only and should not replace")
    report.append("professional medical advice. Always consult your healthcare provider")
    report.append("before making significant dietary changes.")
    
    return "\n".join(report)




# ---------------------------
# Nutrition Analyzer (unchanged)
# ---------------------------
def nutrition_analyzer():
    st.title("🍎 Nutrition Analyzer")
    
    # Health conditions
    health_conditions = {
        "high_blood_pressure": "High Blood Pressure (Hypertension) 🩸",
        "diabetes": "Diabetes/High Blood Sugar 🍯", 
        "heart_disease": "Heart Disease/High Cholesterol ❤️",
        "kidney_disease": "Kidney Disease 🫘",
        "celiac_disease": "Celiac Disease/Gluten Sensitivity 🌾",
        "lactose_intolerance": "Lactose Intolerance 🥛",
        "gout": "Gout/High Uric Acid 🦴",
        "ibs": "Irritable Bowel Syndrome (IBS) 🤢"
    }
    
    selected_conditions = []
    for key, label in health_conditions.items():
        if st.sidebar.checkbox(label, key=f"condition_{key}"):
            selected_conditions.append(key)
    
    if not selected_conditions:
        st.sidebar.warning("⚠️ Select at least one health condition for comprehensive analysis")
    else:
        st.sidebar.success(f"✅ {len(selected_conditions)} condition(s) selected")
    
    # File upload
    st.markdown("### 📸 Upload Nutrition Label Image")
    uploaded_file = st.file_uploader(
        "Choose a nutrition label image...", 
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
        help="Upload a clear, high-resolution image of a nutrition facts label"
    )
    
    if uploaded_file is not None:
        # Create layout columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            # Image metadata
            st.caption(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.caption(f"**Format:** {image.format}")
            st.caption(f"**Mode:** {image.mode}")
        
        with col2:
            st.markdown("### 🔍 Analysis Configuration")
            
            if selected_conditions:
                st.success(f"**Health Conditions:** {len(selected_conditions)} selected")
                condition_names = [health_conditions[cond].split(' (')[0] for cond in selected_conditions]
                st.markdown(f"**Analyzing for:** {', '.join(condition_names)}")
            else:
                st.warning("**No health conditions selected** - Basic nutrition analysis only")
            
            # Analysis button
            if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
                
                try:
                    st.markdown("### 🔄 Analysis Progress")
                    
                    # Run comprehensive analysis
                    all_results, best_extraction, health_analysis = run_nutrition_ocr_with_health_analysis(
                        image, 
                        selected_conditions
                    )
                    
                    # Display results if successful
                    if best_extraction and health_analysis:
                        st.success("🎉 **Analysis Successfully Completed!**")
                        
                        # Extract data from analysis results
                        method_name, nutrition_data, raw_text = best_extraction
                        
                        # Show which method worked best
                        st.info(f"🏆 **Best OCR Method:** {method_name}")
                        
                        # Display all the comprehensive results
                        display_analysis_results(health_analysis, nutrition_data)
                        
                        # Analysis summary
                        with st.expander("📋 Analysis Summary & Technical Details"):
                            st.markdown(f"**🔬 OCR Methods Tested:** {len(all_results)}")
                            st.markdown(f"**🎯 Best Method:** {method_name}")
                            st.markdown(f"**📊 Nutrition Facts Extracted:** {len(nutrition_data.get('nutrition_facts', {}))}")
                            st.markdown(f"**🧪 Ingredients Detected:** {len(nutrition_data.get('ingredients', []))}")
                            st.markdown(f"**🏥 Health Conditions Analyzed:** {len(selected_conditions)}")
                        
                        # Download options
                        st.markdown("### 📁 Download Results")
                        
                        # Create downloadable reports
                        analysis_report = create_text_report(health_analysis, nutrition_data, method_name, selected_conditions)
                        json_data = create_json_report(health_analysis, nutrition_data, method_name)
                        
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="📄 Download Text Report",
                                data=analysis_report,
                                file_name=f"nutrition_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        with col_dl2:
                            st.download_button(
                                label="📊 Download JSON Data",
                                data=json_data,
                                file_name=f"nutrition_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    else:
                        st.error("❌ **Analysis Failed**")
                        st.markdown("""
                        **Possible reasons:**
                        - Image quality too low
                        - Nutrition label not clearly visible
                        - Unsupported image format
                        - OCR models failed to extract text
                        
                        **Tips for better results:**
                        - Use high-resolution images (300+ DPI)
                        - Ensure good lighting and contrast
                        - Crop to just the nutrition facts panel
                        - Avoid blurry or rotated images
                        """)
                
                except Exception as e:
                    st.error(f"❌ **Analysis Error:** {str(e)}")
                    st.markdown("**Troubleshooting:**")
                    st.code(str(e), language="python")
    
    # Information section
    st.markdown("---")
    st.markdown("### 🔬 About This Professional Analysis System")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **🚀 Advanced Features:**
        - **Multiple Image Preprocessing Methods**
        - **Multiple OCR Models** (TrOCR + Tesseract)
        - **Comprehensive Health Database**
        - **8+ Health Condition Analysis**
        - **Professional-Grade Accuracy**
        - **Downloadable Reports**
        """)
    
    with col_info2:
        st.markdown("""
        **🛠️ Technical Stack:**
        - **PyTorch & Transformers** (Microsoft TrOCR)
        - **OpenCV** (Computer Vision)
        - **Advanced Text Processing**
        - **Healthcare Data Analysis**
        - **Streamlit Web Interface**
        """)
    
    st.info("""
    **⚠️ Important Disclaimer:** This tool is designed for educational and informational purposes only. 
    The health analysis provided should not replace professional medical advice, diagnosis, or treatment. 
    Always consult qualified healthcare providers for personalized dietary recommendations.
    """)



# ---------------------------
# Diet Plan Generator (Fixed)
# ---------------------------
def diet_plan_generator():
    st.title("📋 Personalized Diet Plan Generator")
    
    # Check if API key is available
    if not os.getenv('GOOGLE_API_KEY'):
        st.warning("Please set the GOOGLE_API_KEY environment variable to use the diet plan generator")
        st.info("You can get a free API key from: https://aistudio.google.com/app/apikey")
        return
    
    # Collect user requirements
    st.subheader("Personal Information")
    age = st.number_input("Enter your age", min_value=15, max_value=100, value=30)
    gender = st.selectbox("Enter your gender", ["male", "female"])
    weight = st.number_input("Enter your weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Enter your height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    
    st.subheader("Activity & Goals")
    activity_level = st.selectbox("Enter activity level", 
                                 ["sedentary", "light", "moderate", "active", "very active"])
    goal = st.selectbox("Enter your goal", ["weight loss", "muscle gain", "maintenance"])
    
    st.subheader("Food Preferences")
    diet_type = st.selectbox("Enter diet type", 
                            ["vegetarian", "non-vegetarian", "vegan", "gluten-free"])
    allergies = st.text_input("Enter allergies (comma separated, or leave blank if none)")
    dislikes = st.text_input("Enter foods you dislike (comma separated, or leave blank)")
    
    st.subheader("Lifestyle & Health")
    medical_conditions = st.text_input("Enter any medical conditions (comma separated, or leave blank)")
    meal_timing = st.selectbox("Preferred meal timing", ["early eater", "late eater", "flexible"])
    cooking_ability = st.selectbox("Cooking ability", ["beginner", "intermediate", "advanced"])
    
    if st.button("Generate Diet Plan", type="primary", use_container_width=True):
        # Validate inputs
        try:
            if age < 15 or age > 100:
                raise ValueError("Age must be between 15 and 100")
            if weight < 30 or weight > 200:
                raise ValueError("Weight must be between 30 and 200 kg")
            if height < 100 or height > 250:
                raise ValueError("Height must be between 100 and 250 cm")
            if gender not in ["male", "female"]:
                raise ValueError("Gender must be 'male' or 'female'")
            if activity_level not in ["sedentary", "light", "moderate", "active", "very active"]:
                raise ValueError("Invalid activity level")
            if goal not in ["weight loss", "muscle gain", "maintenance"]:
                raise ValueError("Invalid goal")
            
            # Process inputs
            allergies_list = [a.strip().lower() for a in allergies.split(",")] if allergies else []
            dislikes_list = [d.strip().lower() for d in dislikes.split(",")] if dislikes else []
            medical_list = [m.strip().lower() for m in medical_conditions.split(",")] if medical_conditions else []
            
            # Calculate BMR and TDEE
            if gender == "male":
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
            
            activity_factor = {
                "sedentary": 1.2,
                "light": 1.375,
                "moderate": 1.55,
                "active": 1.725,
                "very active": 1.9
            }
            tdee = bmr * activity_factor.get(activity_level, 1.55)

            # Adjust based on goal
            goal_adjustments = {
                "weight loss": 0.85,  # 15% deficit
                "maintenance": 1.0,   # maintain
                "muscle gain": 1.15   # 15% surplus
            }
            tdee = tdee * goal_adjustments.get(goal, 1.0)

            # Macro split based on goal
            if goal == "weight loss":
                macros = {
                    "protein": (0.35 * tdee) / 4,  # 35% protein
                    "carbs": (0.40 * tdee) / 4,    # 40% carbs
                    "fat": (0.25 * tdee) / 9       # 25% fat
                }
            elif goal == "muscle gain":
                macros = {
                    "protein": (0.30 * tdee) / 4,  # 30% protein
                    "carbs": (0.50 * tdee) / 4,    # 50% carbs
                    "fat": (0.20 * tdee) / 9       # 20% fat
                }
            else:  # maintenance
                macros = {
                    "protein": (0.25 * tdee) / 4,  # 25% protein
                    "carbs": (0.45 * tdee) / 4,    # 45% carbs
                    "fat": (0.30 * tdee) / 9       # 30% fat
                }
            
            # Round values
            bmr = round(bmr, 2)
            tdee = round(tdee, 2)
            macros = {k: round(v, 1) for k, v in macros.items()}
            
            # Show nutritional targets
            st.subheader("📊 Your Nutritional Targets")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BMR", f"{bmr} kcal")
            col2.metric("TDEE", f"{tdee} kcal")
            col3.metric("Protein", f"{macros['protein']}g")
            col4.metric("Carbs", f"{macros['carbs']}g")
            st.metric("Fat", f"{macros['fat']}g")
            
            # Generate diet plan using LLM
            with st.spinner("Generating your personalized diet plan..."):
                model = ChatGoogleGenerativeAI(
                    model='models/gemini-2.5-flash',
                    google_api_key=os.getenv('GOOGLE_API_KEY'),
                    temperature=0.7
                )
                
                prompt = f"""
                Create a personalized diet plan based on the following information:
                
                Personal Details:
                - Age: {age}
                - Gender: {gender}
                - Weight: {weight} kg
                - Height: {height} cm
                
                Goals & Activity:
                - Activity Level: {activity_level}
                - Goal: {goal}
                
                Preferences & Restrictions:
                - Diet Type: {diet_type}
                - Allergies: {', '.join(allergies_list) if allergies_list else 'None'}
                - Dislikes: {', '.join(dislikes_list) if dislikes_list else 'None'}
                - Medical Conditions: {', '.join(medical_list) if medical_list else 'None'}
                - Meal Timing Preference: {meal_timing}
                - Cooking Ability: {cooking_ability}
                
                Nutritional Targets:
                - Daily Calories: {tdee} kcal
                - Protein: {macros['protein']}g
                - Carbohydrates: {macros['carbs']}g
                - Fat: {macros['fat']}g
                
                Please create:
                1. A 7-day meal plan with breakfast, lunch, dinner, and 2 snacks
                2. Portion sizes for each meal
                3. Cooking instructions for complex dishes
                4. Alternatives for each meal for variety
                5. Hydration recommendations
                """
                
                diet_response = model.invoke(prompt)
                
                # Generate grocery list
                grocery_prompt = f"""
                Based on the diet plan you just created, generate a comprehensive grocery list 
                organized by categories (produce, proteins, grains, dairy, etc.) with quantities 
                for one week of meals. Consider that the person has: 
                - Diet type: {diet_type}
                - Allergies: {', '.join(allergies_list)}
                - Cooking ability: {cooking_ability}
                """
                
                grocery_response = model.invoke(grocery_prompt)
                
                # Generate progress tracking guidance
                progress_prompt = f"""
                Create a progress tracking guide for someone with these goals: {goal}.
                Include:
                1. Key metrics to track (weight, measurements, etc.)
                2. Recommended frequency of tracking
                3. How to adjust the diet based on progress
                4. Non-scale victories to look for
                5. Timeline for expected results
                """
                
                progress_response = model.invoke(progress_prompt)
                
                # Display results
                st.success("🎉 Your personalized diet plan is ready!")
                
                # Display diet plan
                st.subheader("🍽️ Your 7-Day Diet Plan")
                st.write(diet_response.content)
                
                # Display grocery list
                with st.expander("🛒 Grocery List"):
                    st.write(grocery_response.content)
                
                # Display progress tracking
                with st.expander("📈 Progress Tracking Guide"):
                    st.write(progress_response.content)
                
                # Save results option
                if st.button("💾 Save Results"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"diet_plan_{timestamp}"
                    
                    # Create JSON output
                    json_output = {
                        "personal_info": {
                            "age": age,
                            "gender": gender,
                            "weight": weight,
                            "height": height
                        },
                        "goals": {
                            "activity_level": activity_level,
                            "goal": goal
                        },
                        "preferences": {
                            "diet": diet_type,
                            "allergies": allergies_list,
                            "dislikes": dislikes_list,
                            "cooking_ability": cooking_ability
                        },
                        "medical_conditions": medical_list,
                        "nutritional_targets": {
                            "bmr": bmr,
                            "tdee": tdee,
                            "macros": macros
                        },
                        "generated_on": datetime.now().isoformat()
                    }
                    
                    # Save JSON file
                    with open(f"{filename}.json", "w") as f:
                        json.dump(json_output, f, indent=2)
                    
                    # Save detailed diet plan as text file
                    with open(f"{filename}_diet_plan.txt", "w") as f:
                        f.write("=" * 60 + "\n")
                        f.write("PERSONALIZED DIET PLAN\n")
                        f.write("=" * 60 + "\n\n")
                        
                        f.write("PERSONAL INFORMATION:\n")
                        f.write(f"Age: {age}\n")
                        f.write(f"Gender: {gender}\n")
                        f.write(f"Weight: {weight} kg\n")
                        f.write(f"Height: {height} cm\n\n")
                        
                        f.write("GOALS & ACTIVITY:\n")
                        f.write(f"Activity Level: {activity_level}\n")
                        f.write(f"Goal: {goal}\n\n")
                        
                        f.write("NUTRITIONAL TARGETS:\n")
                        f.write(f"BMR: {bmr} kcal\n")
                        f.write(f"TDEE: {tdee} kcal\n")
                        f.write(f"Protein: {macros['protein']}g\n")
                        f.write(f"Carbohydrates: {macros['carbs']}g\n")
                        f.write(f"Fat: {macros['fat']}g\n\n")
                        
                        f.write("=" * 60 + "\n")
                        f.write("7-DAY MEAL PLAN\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(diet_response.content + "\n\n")
                        
                        f.write("=" * 60 + "\n")
                        f.write("GROCERY LIST\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(grocery_response.content + "\n\n")
                        
                        f.write("=" * 60 + "\n")
                        f.write("PROGRESS TRACKING GUIDE\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(progress_response.content + "\n")
                    
                    st.success(f"Results saved to {filename}.json and {filename}_diet_plan.txt")
        
        except Exception as e:
            st.error(f"Error generating diet plan: {str(e)}")


# # -----------------------
# Enhanced Progress & Analytics
# ---------------------------
def progress_analytics():
    st.title("📊 Progress & Analytics")
    
    # Check if workout data exists
    try:
        df = pd.read_csv("workout_log.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()
        st.info("No workout data available. Start tracking workouts to see your progress!")
    
    if not df.empty:
        # Overall statistics
        st.subheader("📈 Overall Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_workouts = len(df)
        total_reps = df['total_reps'].sum()
        total_calories = df['calories'].sum()
        avg_reps_per_workout = total_reps / total_workouts if total_workouts > 0 else 0
        
        with col1:
            st.metric("Total Workouts", total_workouts)
        
        with col2:
            st.metric("Total Reps", total_reps)
        
        with col3:
            st.metric("Total Calories", f"{total_calories:.0f}")
        
        with col4:
            st.metric("Avg Reps/Workout", f"{avg_reps_per_workout:.1f}")
        
        # Time period selector
        st.subheader("📅 Select Time Period")
        time_period = st.selectbox("Time period", ["Last 7 days", "Last 30 days", "Last 3 months", "Last 6 months", "Last year", "All time"],label_visibility="collapsed")
        
        # Filter data based on selection
        now = datetime.now()
        if time_period == "Last 7 days":
            start_date = now - timedelta(days=7)
            filtered_df = df[df['timestamp'] >= start_date]
        elif time_period == "Last 30 days":
            start_date = now - timedelta(days=30)
            filtered_df = df[df['timestamp'] >= start_date]
        elif time_period == "Last 3 months":
            start_date = now - timedelta(days=90)
            filtered_df = df[df['timestamp'] >= start_date]
        elif time_period == "Last 6 months":
            start_date = now - timedelta(days=180)
            filtered_df = df[df['timestamp'] >= start_date]
        elif time_period == "Last year":
            start_date = now - timedelta(days=365)
            filtered_df = df[df['timestamp'] >= start_date]
        else:
            filtered_df = df
        
        if not filtered_df.empty:
            # Workouts over time chart
            st.subheader("📊 Workouts Over Time")
            workouts_over_time = filtered_df.groupby('date').size().reset_index(name='count')
            fig = px.line(workouts_over_time, x='date', y='count', title='Workouts Per Day')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calories burned over time
            st.subheader("🔥 Calories Burned Over Time")
            calories_over_time = filtered_df.groupby('date')['calories'].sum().reset_index()
            fig = px.line(calories_over_time, x='date', y='calories', title='Calories Burned Per Day')
            st.plotly_chart(fig, use_container_width=True)
            
            # Exercise distribution
            st.subheader("💪 Exercise Distribution")
            exercise_counts = filtered_df['exercise'].value_counts().reset_index()
            exercise_counts.columns = ['exercise', 'count']
            fig = px.pie(exercise_counts, values='count', names='exercise', title='Workouts by Exercise Type')
            st.plotly_chart(fig, use_container_width=True)
            
            # Muscle group distribution
            st.subheader("🏋️ Muscle Group Distribution")
            muscle_group_counts = filtered_df['group'].value_counts().reset_index()
            muscle_group_counts.columns = ['muscle_group', 'count']
            fig = px.bar(muscle_group_counts, x='muscle_group', y='count', title='Workouts by Muscle Group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly progress
            st.subheader("📅 Weekly Progress")
            weekly_data = filtered_df.groupby('week').agg({
                'total_reps': 'sum',
                'calories': 'sum',
                'exercise': 'count'
            }).reset_index()
            weekly_data.columns = ['Week', 'Total Reps', 'Total Calories', 'Workouts']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=weekly_data['Week'], y=weekly_data['Total Reps'], name="Total Reps"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=weekly_data['Week'], y=weekly_data['Workouts'], name="Workouts"),
                secondary_y=True,
            )
            fig.update_layout(title_text="Weekly Progress")
            fig.update_xaxes(title_text="Week")
            fig.update_yaxes(title_text="Total Reps", secondary_y=False)
            fig.update_yaxes(title_text="Workouts", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Achievements
            st.subheader("🏆 Achievements")
            
            # Calculate achievements
            achievements = []
            
            if total_workouts >= 10:
                achievements.append("10 Workouts Completed")
            if total_workouts >= 50:
                achievements.append("50 Workouts Completed")
            if total_workouts >= 100:
                achievements.append("100 Workouts Completed")
            
            if total_reps >= 100:
                achievements.append("100 Reps Completed")
            if total_reps >= 500:
                achievements.append("500 Reps Completed")
            if total_reps >= 1000:
                achievements.append("1000 Reps Completed")
            
            if total_calories >= 1000:
                achievements.append("1000 Calories Burned")
            if total_calories >= 5000:
                achievements.append("5000 Calories Burned")
            if total_calories >= 10000:
                achievements.append("10000 Calories Burned")
            
            # Check for specific exercise achievements
            for exercise, count in df['exercise'].value_counts().items():
                if count >= 10:
                    achievements.append(f"10 {exercise}s Completed")
                if count >= 25:
                    achievements.append(f"25 {exercise}s Completed")
                if count >= 50:
                    achievements.append(f"50 {exercise}s Completed")
            
            # Display achievements
            if achievements:
                for achievement in achievements:
                    st.markdown(f'<span class="achievement-badge">{achievement}</span>', unsafe_allow_html=True)
            else:
                st.info("Keep working out to earn achievements!")
        
        # Workout history table
        st.subheader("📋 Workout History")
        st.dataframe(df[['timestamp', 'group', 'exercise', 'sets_completed', 'target_sets', 'total_reps', 'calories']].sort_values('timestamp', ascending=False))
        
        # Export data
        st.subheader("💾 Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Workout History as CSV",
            data=csv,
            file_name="workout_history.csv",
            mime="text/csv"
        )
    
    # Goals tracking
    st.subheader("🎯 Goal Tracking")
    
    goal = st.selectbox("Set a goal", [
        "Weight Loss", 
        "Muscle Gain", 
        "Maintenance",
        "Endurance Improvement",
        "Flexibility Increase"
    ])
    
    target_date = st.date_input("Target date")
    current_progress = st.slider("Current progress (%)", 0, 100, 25)
    
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {current_progress}%">{current_progress}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"**Goal:** {goal} by {target_date}")
    st.write(f"**Progress:** {current_progress}% complete")
    
    # Add notes section
    st.subheader("📝 Progress Notes")
    notes = st.text_area("Add notes about your progress, challenges, or achievements")
    if st.button("Save Notes"):
        # Save notes to a file (in a real app, you'd use a database)
        with open("progress_notes.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {notes}\n")
        st.success("Notes saved successfully!")

# ---------------------------
# Main App
# ---------------------------
def main():
    # Initialize session state for current section
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "dashboard"
    
    # Navigation
    selected_section = navigation()
    
    # Update session state if navigation changed
    if selected_section != st.session_state.current_section:
        st.session_state.current_section = selected_section
        st.rerun()
    
    # Display the selected section
    if st.session_state.current_section == "dashboard":
        show_dashboard()
    elif st.session_state.current_section == "workout":
        workout_tracker()
    elif st.session_state.current_section == "nutrition":
        nutrition_analyzer()
    elif st.session_state.current_section == "diet":
        diet_plan_generator()
    elif st.session_state.current_section == "progress":
        progress_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Fitness Pro - Complete Health & Fitness Suite</p>
        <p>💪 Track workouts • 🍎 Analyze nutrition • 📋 Plan diets • 📊 Monitor progress</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Load environment variables for diet plan generator
    load_dotenv()
    
    # Run the main app
    main()

# Create the workflow for diet plan generator
workflow = StateGraph(DietChart)

# Add nodes
workflow.add_node("collect", collect_user_requirements)
workflow.add_node("validate", validate_inputs)
workflow.add_node("calculate", calculate_macros)
workflow.add_node("generate", generate_diet_chart)
workflow.add_node("save", save_results)

# Add edges
workflow.add_edge(START, "collect")
workflow.add_edge("collect", "validate")
workflow.add_edge("validate", "calculate")
workflow.add_edge("calculate", "generate")
workflow.add_edge("generate", "save")
workflow.add_edge("save", END)

# Compile the workflow
app = workflow.compile()