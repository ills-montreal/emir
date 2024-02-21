import os


def get_model_path(MODEL_PATH = "backbone_pretrained_models", models = []):
    MODELS = {}
    # For every directory in the folder
    for model_name in os.listdir(MODEL_PATH):
        if model_name in models:
            # For every file in the directory
            for file_name in os.listdir(os.path.join(MODEL_PATH, model_name)):
                # If the file is a .pth file
                if file_name.endswith(".pth"):
                    MODELS[model_name] = os.path.join(MODEL_PATH, model_name, file_name)
    MODELS["Not-trained"] = ""
    return MODELS