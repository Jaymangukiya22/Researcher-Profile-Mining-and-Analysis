import pandas as pd

# Create a sample DataFrame with researcher names
df = pd.DataFrame({
    'Researcher Name': [
        'Siqi Sun',
        'Christos Christodoulopoulos',
        'Shuyang Cao',
        'Ivan Habernal',
        'Hongfei Xu',
        'Aarne Talman',
        'Alessandro Suglia',
        'Yixin Cao',
        'Bo Zheng',
        'Zhiqing Sun'
        'Yunfang Wu',
        'Monjoy Saha'
        'Bashar Alhafni',
        'Chinnadhurai Sankar',
        'Jun Xu'
        'Reinald Kim Amplayo',
        'Yilun Zhu',
        'Matthew Mulholland',
        'Qinlan Shen',
    ]
})

# Save to Excel file
df.to_excel('researchers.xlsx', index=False)
print("Created researchers.xlsx template with the provided researcher names.") 