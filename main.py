import requests
import streamlit as st

API_URL = 'https://arpy8-skin-lesion-classification-api.hf.space/predict'

"# Skin Lesion Classification"

with st.container():
    uploaded_file = st.file_uploader("Upload a file", type=["jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file)
        
    with st.columns([1, 2, 1])[0]:
        submit_button = st.button("Predict", use_container_width=True)
    
    if submit_button:
        if uploaded_file is not None:
            try:
                files = {'file': uploaded_file}
                headers = {'accept': 'application/json'}
                response = requests.post(API_URL, headers=headers, files=files)
                response.raise_for_status()
                result = response.json()
                if round(result['confidence'], 4)*100 < 75:
                    st.warning("The image might not be of a skin lesion.")

                st.info(f"""
                **Label**: {result['label']}
                
                **Probability**: {round(result['confidence'], 4)*100}%
                """)
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image first.")