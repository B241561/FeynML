import joblib


def load_model(filepath):
    """
    Loads a trained ML model from disk.
    Supports .pkl (pickle) and .joblib formats.

    Args:
        filepath (str): Path to the model file.

    Returns:
        The loaded model object.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if not (filepath.endswith('.pkl') or filepath.endswith('.joblib')):
        raise ValueError("Unsupported model format. Please upload a .pkl or .joblib file.")

    # joblib.load() handles both joblib-saved AND plain pickle-saved files —
    # it's a superset of pickle, so this works regardless of which
    # library the user's .pkl was originally saved with.
    model = joblib.load(filepath)

    return model