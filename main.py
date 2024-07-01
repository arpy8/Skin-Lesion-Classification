import requests
import numpy as np
from PIL import Image
import streamlit as st

API_URL = 'http://127.0.0.1:8000/process_image'
# API_URL = 'https://arpy8-skin-lesion-obj-detection-api.hf.space/process_image'
# API_URL2 = 'https://arpy8-skin-lesion-classification-api.hf.space/predict'


st.set_page_config(page_title="Skin Lesion Classification", layout="wide")

"# Skin Lesion Classification"


# def clf_backup(uploaded_file):
#     files = {'file': uploaded_file}
#     headers = {'accept': 'application/json'}
#     response = requests.post(API_URL2, headers=headers, files=files)
#     response.raise_for_status()
#     result = response.json()
    
#     return result


with st.container():
    uploaded_file = st.file_uploader("Upload a file", type=["jpeg", "png"])
    
    # if uploaded_file is not None:
    #     st.image(uploaded_file)
        
    with st.columns([1, 2, 1])[0]:
        submit_button = st.button("Predict", use_container_width=True)
    
    if submit_button:
        if uploaded_file is not None:
            if uploaded_file not in st.session_state:
                st.session_state["uploaded_file"] = uploaded_file
            
            try:
                files = {'file': uploaded_file}
                headers = {'accept': 'application/json'}
                response = requests.post(API_URL, headers=headers, files=files)
                response.raise_for_status()
                
                
                left, right = st.columns([2, 1])
                with left:
                    if response.status_code == 200:
                        image = response.json()['image']['annotated_image']
                        st.image(Image.fromarray(np.array(image).astype(np.uint8)))

                    # if len(response.json()['classification'].keys()) == 1:
                    # else:
                    result = response.json()['classification']
                # result = clf_backup(st.session_state["uploaded_file"])
                
                # if round(result['confidence'], 4)*100 < 75:
                #     st.warning("The image might not be of a skin lesion.")

                # st.info(f"""
                # **Label**: {result['label']}
                
                # **Description**: {result['description']}
                
                # **Probability**: {round(result['confidence'], 4)*100}%
                # """)
                # st.link_button("Know More", result['link'])
                with right:
                    st.write(response.json()['classification'])
                        
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image first.")