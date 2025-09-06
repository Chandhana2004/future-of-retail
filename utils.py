from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import pandas as pd
import re
import spacy
import difflib

# Load model and dataset
model = joblib.load('model.pkl')
data = pd.read_csv('bigmart_data.csv')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def predict_sales(features):
    feature_names = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                     'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size',
                     'Outlet_Location_Type', 'Outlet_Type']
    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_df)[0]
    return prediction

def chatbot_response(msg):
    msg = msg.lower().strip()
    doc = nlp(msg)

    # Helper function to match item type more flexibly
    def find_matching_category(user_input, categories):
        user_input = user_input.lower().strip()

        synonyms = {
            "snacks": "Snack Foods",
            "beverages": "Soft Drinks",
            "drinks": "Soft Drinks",
            "soft drinks": "Soft Drinks",
            "dairy": "Dairy",
            "milk": "Dairy",
            "frozen": "Frozen Foods",
            "vegetables": "Fruits and Vegetables",
            "fruits": "Fruits and Vegetables",
            "health": "Health and Hygiene",
            "hygiene": "Health and Hygiene",
            "household": "Household",
            "meat": "Meat",
            "sea food": "Seafood",
            "seafood": "Seafood",
            "staples": "Starchy Foods",
            "baking": "Baking Goods",
            "bread": "Breads",
        }

        # Use synonym if available
        if user_input in synonyms:
            user_input = synonyms[user_input].lower()

        # Direct or partial match
        for cat in categories:
            cat_lower = cat.lower()
            if user_input in cat_lower or cat_lower in user_input:
                return cat

        # Fuzzy match
        matches = difflib.get_close_matches(user_input, [c.lower() for c in categories], n=1, cutoff=0.6)
        if matches:
            for cat in categories:
                if matches[0] == cat.lower():
                    return cat

        return None

    # Greeting
    if any(token.text in ["hi", "hello", "hey", "hii"] for token in doc):
        return "Hi! Ask me about Big Mart sales, products, or outlet details."

    # Average sales overall or for category
    avg_match = re.search(r'average sales(?: of ([\w\s]+))?', msg)
    if avg_match:
        cat = avg_match.group(1)
        if not cat:
            avg = data['Item_Outlet_Sales'].mean()
            return f"The average sales overall is {avg:.2f}."
        else:
            match_cat = find_matching_category(cat.strip(), data['Item_Type'].unique())
            if match_cat:
                avg = data[data['Item_Type'] == match_cat]['Item_Outlet_Sales'].mean()
                return f"The average sales of '{match_cat}' is {avg:.2f}."
            else:
                return f"No data found for item type '{cat.strip()}'."            

    # Total items
    if "how many" in msg and "item" in msg:
        total = data['Item_Identifier'].nunique()
        return f"There are {total} unique items in the dataset."

    # Total outlets
    if "how many" in msg and "outlet" in msg:
        total = data['Outlet_Identifier'].nunique()
        return f"There are {total} unique outlets."

    # Highest sales outlet
    if "highest sales outlet" in msg or "best outlet" in msg:
        grp = data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()
        return f"The highest sales outlet is {grp.idxmax()} with total sales of {grp.max():.2f}."

    # Product availability
    match = re.search(r'is ([\w\d_]+) available in ([\w\d_]+)', msg)
    if match:
        item, outlet = match.group(1), match.group(2)
        filt = data[
            (data['Item_Identifier'].str.lower() == item.lower()) & 
            (data['Outlet_Identifier'].str.lower() == outlet.lower())
        ]
        return f"Yes, '{item}' is available in '{outlet}'." if not filt.empty else f"No, '{item}' is not available in '{outlet}'."

    # Total sales of item type
    match = re.search(r'total sales of ([\w\s]+)', msg)
    if match:
        raw_cat = match.group(1).strip()
        match_cat = find_matching_category(raw_cat, data['Item_Type'].unique())
        if match_cat:
            filt = data[data['Item_Type'] == match_cat]
            total = filt['Item_Outlet_Sales'].sum()
            return f"Total sales of '{match_cat}': {total:.2f}"
        else:
            return f"No data found for item type '{raw_cat}'."

  # Sales percentage of item type, outlet or item ID
    match = re.search(r'sales percentage of ([\w\d\s_]+)', msg)
    if match:
        entity = match.group(1).strip().lower()
        total_sales = data['Item_Outlet_Sales'].sum()

        # Check if it's an item type
        match_cat = find_matching_category(entity, data['Item_Type'].unique())
        if match_cat:
            cat_sales = data[data['Item_Type'] == match_cat]['Item_Outlet_Sales'].sum()
            percent = (cat_sales / total_sales) * 100
            return f"Sales percentage of '{match_cat}' is {percent:.2f}% of total sales."

        # Check if it's an outlet
        outlet_ids = data['Outlet_Identifier'].str.lower().unique()
        if entity in outlet_ids:
            outlet_sales = data[data['Outlet_Identifier'].str.lower() == entity]['Item_Outlet_Sales'].sum()
            percent = (outlet_sales / total_sales) * 100
            return f"Sales percentage of outlet '{entity.upper()}' is {percent:.2f}% of total sales."

        # Check if it's an item ID
        item_ids = data['Item_Identifier'].str.lower().unique()
        if entity in item_ids:
            item_sales = data[data['Item_Identifier'].str.lower() == entity]['Item_Outlet_Sales'].sum()
            percent = (item_sales / total_sales) * 100
            return f"Sales percentage of item '{entity.upper()}' is {percent:.5f}% of total sales."

        return f"No data found for '{entity}'."
    
    # Thank you
    if any(phrase in msg for phrase in ["thank you", "thanks", "thankyou", "thx"]):
        return "You're welcome! Let me know if you have more questions."

    # Goodbye
    if any(phrase in msg for phrase in ["bye", "goodbye", "exit", "quit"]):
        return "Goodbye! Have a great day!"

    # Sales rate or percentage
    if "sales rate" in msg or "sales percentage" in msg:
        return "Could you please clarify what you mean by sales rate or percentage? Are you referring to growth, average, or a specific item's contribution?"

    return "Sorry, I couldn't understand your question. Try asking about average sales, item availability, or outlet details."