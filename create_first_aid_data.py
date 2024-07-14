import pandas as pd
import csv

# Data for first aid advice with steps
data = {
    "advice_id": [1, 2, 3],
    "advice_text": [
        "Apply pressure to the wound to stop bleeding.",
        "Rinse the burn with cool water for 10-20 minutes.",
        "Perform CPR if the person is not breathing."
    ],
    "steps": [
        "1. Apply firm pressure directly to the wound with a clean cloth or bandage.\n"
        "2. Maintain pressure until bleeding stops or medical help arrives.",
        
        "1. Hold the burn under cool running water for at least 10 minutes.\n"
        "2. Gently pat the burn dry with a clean cloth.\n"
        "3. Cover the burn with a sterile dressing.",
        
        "1. Check for responsiveness and breathing.\n"
        "2. If not breathing, call emergency services (911 in the US).\n"
        "3. Perform chest compressions at a rate of 100-120 per minute until help arrives."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to CSV
df.to_csv('first_aid_data.csv', index=False, line_terminator='\n', quoting=csv.QUOTE_MINIMAL)

print("first_aid_data.csv file has been created successfully.")
