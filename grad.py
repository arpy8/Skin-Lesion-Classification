import requests
import numpy as np
from PIL import Image
import gradio as gr

API_URL = 'https://arpy8-skin-lesion-obj-detection-api.hf.space/process_image'
API_URL2 = 'https://arpy8-skin-lesion-classification-api.hf.space/predict'

def clf_backup(image):
    files = {'file': image}
    headers = {'accept': 'application/json'}
    response = requests.post(API_URL2, headers=headers, files=files)
    response.raise_for_status()
    result = response.json()
    
    return result

def process_image(image):
    if image is not None:
        try:
            files = {'file': image}
            headers = {'accept': 'application/json'}
            response = requests.post(API_URL, headers=headers, files=files)
            response.raise_for_status()
            
            if response.status_code == 200:
                annotated_image = response.json()['image']['annotated_image']
                annotated_image = Image.fromarray(np.array(annotated_image).astype(np.uint8))

                result = clf_backup(image)
                
                label = result['label']
                description = result['description']
                confidence = round(result['confidence'], 4) * 100
                link = result['link']
                
                if confidence < 75:
                    warning_message = "The image might not be of a skin lesion."
                else:
                    warning_message = None

                return annotated_image, label, description, confidence, link, warning_message
        except requests.exceptions.HTTPError as e:
            return None, None, None, None, None, f"HTTP Error: {e}"
        except Exception as e:
            return None, None, None, None, None, f"An error occurred: {e}"
    else:
        return None, None, None, None, None, "Please upload an image first."

def interface(image):
    annotated_image, label, description, confidence, link, warning_message = process_image(image)
    return annotated_image, label, description, confidence, link, warning_message

    results = {
        "Annotated Image": annotated_image,
        "Label": label,
        "Description": description,
        "Confidence (%)": confidence,
        "Link": link,
        "Warning": warning_message
    }
    return results
    
iface = gr.Interface(
    fn=interface,
    inputs=gr.inputs.Image(type="filepath", label="Upload an image"),
    outputs=[
        gr.outputs.Image(type="pil", label="Annotated Image"),
        gr.outputs.Textbox(label="Label"),
        gr.outputs.Textbox(label="Description"),
        gr.outputs.Textbox(label="Confidence (%)"),
        gr.outputs.Textbox(label="Link"),
        gr.outputs.Textbox(label="Warning")
    ],
    title="Skin Lesion Classification",
    description="Upload a skin lesion image to get its classification.",
)

iface.launch()