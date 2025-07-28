import json
from pathlib import Path
from app.utils import load_persona

def test_persona_loading():
    # Example test using Challenge_1b input
    persona_path = Path('dataset/Challenge_1b/Collection 1/challenge1b_input.json')
    persona, job = load_persona(persona_path)
    assert 'persona' in str(persona) or job != '', "Persona or job missing"
