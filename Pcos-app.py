import streamlit as st
import pandas as pd
from joblib import load
from PIL import Image
import os

# โหลดรูปภาพแบบตรวจสอบไฟล์
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ภาพ: {path}")
        return None

HairG = load_image("hairgrowP.jpg")
Skindarken = load_image("skin darkenP.jpg")

st.title("🩺 Assessing the Risk of Polycystic Ovary Syndrome (PCOS)")
st.markdown("""
## แอปประเมินความเสี่ยงโรคถุงน้ำรังไข่หลายใบ  
โปรแกรมนี้ช่วยประเมินความเสี่ยงเบื้องต้นจากข้อมูลสุขภาพของคุณ  
⚠️ **คำเตือน:** ผลลัพธ์นี้ไม่ใช่การวินิจฉัยจากแพทย์
""")

st.sidebar.header("กรอกข้อมูลเพื่อประเมินความเสี่ยง")

# ฟังก์ชันรับข้อมูลจากผู้ใช้
def user_input_features():
    Age = st.sidebar.slider('อายุ (ปี)', 0, 100, 22)
    Weight = st.sidebar.slider('น้ำหนัก (กิโลกรัม)', 0, 150, 79)
    Cycle = st.sidebar.slider('ประจำเดือนมากี่วัน', 0, 31, 7)
    CycleLength = st.sidebar.slider('ระยะห่างของรอบเดือน (วัน)', 0, 60, 16)
    hairGrowth = st.sidebar.slider('ขนเพิ่มขึ้นหรือไม่', 0, 1, 1)
    if HairG:
        st.sidebar.image(HairG, caption="Ferriman Hair Growth Chart", use_container_width=True)

    SkinDarkening = st.sidebar.slider('ผิวดำคล้ำตามข้อหรือไม่', 0, 1, 0)
    if Skindarken:
        st.sidebar.image(Skindarken, caption="จุดสังเกตผิวคล้ำ", use_container_width=True)

    Pimples = st.sidebar.slider('สิวเพิ่มขึ้นหรือไม่', 0, 1, 1)
    Fastfood = st.sidebar.slider('ทานอาหารไขมันสูงบ่อยหรือไม่', 0, 1, 0)
    FollicleL = st.sidebar.slider('รูขุมขนกว้างด้านซ้าย', 0, 1, 1)
    FollicleR = st.sidebar.slider('รูขุมขนกว้างด้านขวา', 0, 1, 1)
    WeightGain = st.sidebar.slider('น้ำหนักเพิ่มขึ้นเร็วหรือไม่', 0, 1, 1)

    data = {
        'Age (yrs)': Age,
        'Weight (Kg)': Weight,
        'Cycle(R/I)': Cycle,
        'Cycle length(days)': CycleLength,
        'hair growth(Y/N)': hairGrowth,
        'Skin darkening (Y/N)': SkinDarkening,
        'Pimples(Y/N)': Pimples,
        'Fast food (Y/N)': Fastfood,
        'Follicle No. (L)': FollicleL,
        'Follicle No. (R)': FollicleR,
        'Weight gain(Y/N)': WeightGain
    }

    return pd.DataFrame(data, index=[0])

# โหลดโมเดล
model_path = "PcosApp.joblib"
if not os.path.exists(model_path):
    st.error(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
    st.stop()

app = load(model_path)

df = user_input_features()
st.subheader("ข้อมูลที่คุณกรอก")
st.write(df)

if st.button("ทำการประเมินความเสี่ยง"):
    prediction = app.predict(df)
    prediction_proba = app.predict_proba(df)

    if prediction[0] == 1:
        st.error("**ผลลัพธ์:** มีความเสี่ยงสูง ควรพบแพทย์เพื่อตรวจเพิ่มเติม")
    else:
        st.success("**ผลลัพธ์:** ความเสี่ยงต่ำ")

    st.subheader("เปอร์เซ็นต์ความเสี่ยง")
    st.write({
        "โอกาสเสี่ยงต่ำ": f"{prediction_proba[0][0]*100:.2f}%",
        "โอกาสเสี่ยงสูง": f"{prediction_proba[0][1]*100:.2f}%"
    })
