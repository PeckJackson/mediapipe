from pathlib import Path


def get_face_model_path(model_name, file_type="task"):
    project_root = Path(__file__).parent.parent.parent
    
    model_path = project_root / "models" / f"{model_name}.{file_type}"

    if model_path.exists():
        return model_path
    else:
        raise Exception(f"Model Does Not Exist: {model_name}")