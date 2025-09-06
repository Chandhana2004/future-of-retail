import pandas as pd
import re

df = pd.read_csv('bigmart_data.csv')

# Lowercase important columns once
df['Item_Type'] = df['Item_Type'].str.lower()
df['Item_Identifier'] = df['Item_Identifier'].str.lower()
df['Outlet_Identifier'] = df['Outlet_Identifier'].str.lower()

def chatbot_response(msg):
    msg = msg.lower()
    
    if any(greet in msg for greet in ['hi', 'hello']):
        return "Hi! Ask me about Big Mart sales or product availability."

    avg = re.search(r'average sales of ([\w\s]+)', msg)
    if avg:
        cat = avg.group(1).strip()
        filt = df[df['Item_Type'] == cat]
        if filt.empty:
            return f"No data for '{cat}'"
        return f"Avg sales of '{cat}': {filt['Item_Outlet_Sales'].mean():.2f}"

    if 'highest sales outlet' in msg:
        grp = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()
        return f"Best outlet: {grp.idxmax()} with sales {grp.max():.2f}"

    total = re.search(r'total sales of ([\w\s]+)', msg)
    if total:
        cat = total.group(1).strip()
        filt = df[df['Item_Type'] == cat]
        if filt.empty:
            return f"No data for '{cat}'"
        return f"Total sales of '{cat}': {filt['Item_Outlet_Sales'].sum():.2f}"

    available = re.search(r'is ([\w\d_]+) available in ([\w\d_]+)', msg)
    if available:
        item, outlet = available.group(1).strip(), available.group(2).strip()
        filt = df[(df['Item_Identifier'] == item) & (df['Outlet_Identifier'] == outlet)]
        if filt.empty:
            return f"'{item}' is not available in '{outlet}'"
        return f"Yes, '{item}' is available in '{outlet}'"

    return "Sorry, I didn't understand. Ask about sales, products, outlets, or availability."
