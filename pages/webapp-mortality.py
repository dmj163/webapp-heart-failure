import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 页面内容设置
# 页面名称
st.set_page_config(page_title="Mortality", layout="wide")
# 标题
st.title('The machine-learning model to predict mortality for heart failure')

st.markdown('_This is a webApp to predict the risk of **death** based on several features that you can see in the sidebar.\
         Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction._')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)


def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"
@st.cache
def predict_quality(model, df):
    y_pred = model.predict_proba(df)
    return y_pred[:, 1]


# 导入模型
model = joblib.load('save/lgb_death_less.pkl')

st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
age = st.sidebar.slider(label='age', min_value=18,
                                  max_value=120,
                                  value=70,
                                  step=1)


SBP = st.sidebar.slider(label='SBP', min_value=1,
                                max_value=200,
                                value=120,
                                step=1)

DBP = st.sidebar.slider(label='DBP', min_value=1,
                                   max_value=200,
                                   value=80,
                                   step=1)


NTproBNP = st.sidebar.number_input(label='NT-proBNP', min_value=0.00,
                       max_value=100000.00,
                       value=30000.00,
                       step=1.00)



Cholinesterase = st.sidebar.number_input(label='Cholinesterase', min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=0.01)

ALB = st.sidebar.number_input(label='ALB', min_value=1.0,
                            max_value=100.0,
                            value=1.0,
                            step=0.1)

Fibrinogen = st.sidebar.number_input(label='Fibrinogen', min_value=0.0,
                            max_value=20.0,
                            value=0.0,
                            step=0.01)

TroponinT = st.sidebar.number_input(label='Troponin T', min_value=0.0,
                            max_value=1000.0,
                            value=0.0,
                            step=0.001)

AST = st.sidebar.number_input(label='AST', min_value=0.0,
                            max_value=10000.0,
                            value=0.0,
                            step=0.1)

Ca = st.sidebar.number_input(label='Ca', min_value=0.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.1)


features = {'age': age,
            'SBP': SBP,
            'DBP': DBP,
            'NT-proBNP': NTproBNP,
            'Cholinesterase': Cholinesterase,
            'ALB': ALB,
            'Fibrinogen':Fibrinogen,
            'Troponin T': TroponinT,
            'Ca': Ca,
            'AST': AST
}

features_df = pd.DataFrame([features])
#显示输入的特征
st.table(features_df)

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of mortality:")
    st.success(round(prediction[0], 2))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value[1], shap_values[1], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')
    st.image('test_shap.png', caption='Individual prediction explaination', use_column_width=True)


