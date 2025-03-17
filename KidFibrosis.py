import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
model = joblib.load('XGB.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ["HBsAg","HBeAb","GGT","AKP"]

# 创建Streamlit应用
st.title('KidFibrosis')
st.write('This application can help you predict the probability of significant liver fibrosis in children with CHB')
# 创建用户输入控件
HBsAg = st.number_input('HBsAg (IU/mL)', min_value=0.000, max_value=60000.000, value=0.000, step=1.00, key='HBsAg')
HBeAb = st.selectbox('HBeAb', options=[0, 1], format_func=lambda x: 'Negative' if x == 0 else 'Positive')
GGT = st.number_input('GGT (IU/L)', min_value=0.00, max_value=200.00, value=0.00, step=1.00, key='GGT')
AKP = st.number_input('AKP (IU/L)', min_value=0.00, max_value=1000.00, value=0.00, step=1.00, key='AKP')


# 计算log10(HBsAg)值
log10_HBsAg = np.log10(HBsAg) if HBsAg > 0 else 0

feature_values = [log10_HBsAg, HBeAb, GGT,AKP]
features = np.array([feature_values])

# 创建一个按钮进行预测
if st.button('Predict'):
    # 检查是否所有输入都已经提供
    if HBsAg == 0.00 or GGT == 0.00 or AKP == 0.00:
        st.write("Please fill in all fields")
    else:
    # 获取用户输入并创建数据框
     user_data = pd.DataFrame({
        'HBsAg': [log10_HBsAg],
        'HBeAb': [HBeAb],
        'GGT': [GGT],
        'AKP': [AKP]
    })
    
    # 对用户输入的数据进行标准化处理
    numerical_features = user_data[['HBsAg', 'GGT', 'AKP']]  # 只对数值变量标准化
    scaled_features = scaler.transform(numerical_features)

    user_data_scaled = np.column_stack((scaled_features, user_data['HBeAb'].values))
    
    # 进行预测
    prediction_prob = model.predict_proba(user_data_scaled)[0, 1]
    
    # 显示预测结果
    st.write(f'The probability of significant liver fibrosis is: {prediction_prob * 100:.2f}%')
    # Generate advice based on prediction results    
    if prediction_prob >=0.62:        
        advice = (            f'According to our model, CHB children with a predicted probability greater than 62% have a high risk of significant liver fibrosis. '            
                              f'The model predicts that the probability of having significant liver fibrosis is {prediction_prob * 100:.2f}%. '           
                              'While this is just an estimate, it suggests that this patient may be at significant risk. '           
                              'I recommend that this patient undergo a liver biopsy as soon as possible for further evaluation and '         
                              'to ensure accurate diagnosis and necessary treatment.' )    
    else:        
        advice = (            f'According to our model, CHB children with a predicted probability greater than 62% have a high risk of significant liver fibrosis. '       
                              f'The model predicts that the probability of having significant liver fibrosis is {prediction_prob * 100:.2f}%. '           
                              'However, maintaining a healthy lifestyle is still very important.'            
                              'I recommend regular check-ups to monitor his or her liver health, '            
                              'and to seek medical advice promptly if this child experience any symptoms.'        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    st.subheader("SHAP Force Plot")
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
