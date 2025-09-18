import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph

load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(
    model='models/gemini-2.5-flash',
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.7
)

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

def collect_user_requirements(state: DietChart) -> DietChart:
    """Collect comprehensive user information for diet planning"""
    print("\n=== Personal Information ===")
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (male/female): ").strip().lower()
    weight = float(input("Enter your weight (kg): "))
    height = float(input("Enter your height (cm): "))
    
    print("\n=== Activity & Goals ===")
    activity_level = input("Enter activity level (sedentary/light/moderate/active/very active): ").strip().lower()
    goal = input("Enter your goal (weight loss/muscle gain/maintenance): ").strip().lower()
    
    print("\n=== Food Preferences ===")
    diet_type = input("Enter diet type (vegetarian/non-vegetarian/vegan/gluten-free): ").strip().lower()
    allergies = input("Enter allergies (comma separated, or leave blank if none): ").strip()
    dislikes = input("Enter foods you dislike (comma separated, or leave blank): ").strip()
    
    print("\n=== Lifestyle & Health ===")
    medical_conditions = input("Enter any medical conditions (comma separated, or leave blank): ").strip()
    meal_timing = input("Preferred meal timing (early eater/late eater/flexible): ").strip().lower()
    cooking_ability = input("Cooking ability (beginner/intermediate/advanced): ").strip().lower()
    
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

def save_results(state: DietChart) -> DietChart:
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
    
    print(f"\nResults saved to {filename}.json and {filename}_diet_plan.txt")
    return state

# Create the workflow
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

# Run the application
if __name__ == "__main__":
    print("=== Personalized Diet Plan Generator ===")
    print("Please provide the following information to create your custom diet plan\n")
    
    final_state = app.invoke({})
    
    print("\n=== Diet Plan Generation Complete ===")
    print(f"\nYour personalized diet plan has been created with:")
    print(f"- Daily calorie target: {final_state['tdee']:.0f} kcal")
    print(f"- Macronutrients: {final_state['macros']['protein']:.0f}g protein, "
          f"{final_state['macros']['carbs']:.0f}g carbs, "
          f"{final_state['macros']['fat']:.0f}g fat")