# 🤖 AI Code Generation Prompt - Selective Face Anonymization

## Quick Context

I have a face anonymization system with:

- YOLOv11 face detection (working)
- Multiple anonymization modes: blur, pixelate, silhouette (working)
- Real-time camera processing (working)

## What I Need

Implement **selective face anonymization** with family member registration:

1. Register family members (capture faces, extract embeddings, save to DB)
2. During real-time processing: detect → recognize → selectively anonymize
3. Two modes: anonymize ONLY family OR anonymize ONLY strangers

## Code Structure Needed

### 1. Face Recognition Module (`src/face_recognition.py`)

```python
class FaceRecognizer:
    """Extract 512-d embeddings using DeepFace with Facenet512 model."""

    def extract_embedding(face_image: np.ndarray) -> np.ndarray:
        # Use DeepFace.represent() with model_name="Facenet512"
        # Return 512-dimensional vector

    def cosine_similarity(emb1, emb2) -> float:
        # Return similarity score 0-1
```

### 2. Family Database (`src/family_database.py`)

```python
class FamilyDatabase:
    """JSON-based storage for family member embeddings."""

    def add_member(name: str, embeddings: List[np.ndarray]) -> str:
        # Save to data/family_members.json
        # Structure: {member_id, name, embeddings(list of 512-d vectors), timestamp}

    def match(query_embedding, threshold=0.6) -> Optional[Tuple[id, name, score]]:
        # Compare with all stored embeddings using cosine similarity
        # Return best match if score > threshold
```

### 3. Selective Anonymizer (`src/selective_anonymizer.py`)

```python
class SelectiveAnonymizer:
    def process_frame(frame, anonymize_target="family"):
        # 1. Detect faces (use existing FaceDetector)
        # 2. Extract embeddings for each face
        # 3. Match against database
        # 4. If anonymize_target=="family": blur matched faces
        #    If anonymize_target=="stranger": blur non-matched faces
        # 5. Use existing anonymize_faces() function
```

### 4. Registration Tool (`registration_tool.py`)

```python
class RegistrationTool:
    def register_from_camera(name, num_captures=10):
        # Open camera with OpenCV
        # Show live preview with face detection
        # Press SPACE to capture (only if exactly 1 face detected)
        # Collect 10 embeddings from different angles
        # Save to database
        # Show: "Captured: X/10" status
```

## Existing Code to Integrate With

**Face Detection (already working):**

```python
from src.detector import FaceDetector
detector = FaceDetector(model_path="models/yolov11n-face.pt")
detections = detector.detect(frame)  # Returns [(box, conf), ...]
# box format: [x, y, w, h]
```

**Anonymization (already working):**

```python
from src.privacy import anonymize_faces
anonymized = anonymize_faces(
    frame,
    boxes,  # List of [x, y, w, h]
    mode="blur"  # or "pixelate", "silhouette", etc.
)
```

## Algorithm Flow

```
REGISTRATION:
  Camera → Capture 10 photos → Extract embeddings → Save to JSON

RECOGNITION:
  Frame → Detect faces → Extract embeddings →
  For each face:
    Compare with stored embeddings using cosine similarity
    If similarity > 0.6: match found

SELECTIVE ANONYMIZATION:
  If mode="family":
    Anonymize matched faces, keep strangers visible
  If mode="stranger":
    Anonymize non-matched faces, keep family visible
```

## Technical Requirements

- **Library**: DeepFace with Facenet512 model
- **Embedding size**: 512 dimensions
- **Similarity metric**: Cosine similarity
- **Threshold**: 0.6 (adjustable via parameter)
- **Database**: JSON file at `data/family_members.json`
- **Performance**: Should run at >10 FPS with 2-3 faces

## Command-line Interface

```bash
# Register family member
python registration_tool.py --register --name "Nguyen Van A"

# Run selective anonymization
python src/run_phone_cam.py --selective-mode family --show

# Modes:
#   --selective-mode disabled  (default: anonymize all faces)
#   --selective-mode family    (anonymize only registered family members)
#   --selective-mode stranger  (anonymize only strangers)
```

## Sample Output Format

```python
# SelectiveAnonymizer.process_frame() should return:
{
    "frame": anonymized_frame,  # np.ndarray
    "detections": [
        {
            "box": [x, y, w, h],
            "confidence": 0.95,
            "matched": True,
            "member_name": "Nguyen Van A",
            "match_score": 0.87,
            "anonymized": True
        },
        {
            "box": [x2, y2, w2, h2],
            "confidence": 0.92,
            "matched": False,
            "member_name": None,
            "match_score": None,
            "anonymized": False
        }
    ]
}
```

## Error Handling

```python
# Handle these cases:
- No face detected during registration → Show warning, retry
- Multiple faces during registration → Show "Only 1 face allowed"
- Embedding extraction fails → Log error, skip frame
- Database file not found → Create new empty database
- Invalid threshold (not 0-1) → Raise ValueError
```

## Testing Checklist

- [ ] Can register person with 10 photos
- [ ] Database saves correctly to JSON
- [ ] Matching achieves >90% accuracy
- [ ] Family mode: only family faces are blurred
- [ ] Stranger mode: only stranger faces are blurred
- [ ] FPS stays above 10 with 2-3 faces
- [ ] Registration UI shows clear instructions

---

## Generate These 4 Files:

1. **`src/face_recognition.py`** - FaceRecognizer class with DeepFace
2. **`src/family_database.py`** - JSON database with CRUD operations
3. **`src/selective_anonymizer.py`** - Selective anonymization logic
4. **`registration_tool.py`** - Interactive registration tool with OpenCV UI

Include detailed docstrings, type hints, and error handling.

---

**Expected time to implement:** 2-3 hours  
**Key challenge:** Optimizing FPS while maintaining accuracy  
**Main library:** `pip install deepface tf-keras`
