# AI Implementation Prompt: Selective Face Anonymization System

## 🎯 Project Context

You are working on a **Face Anonymization System** for a computer vision thesis project. The system currently provides **universal face anonymization** (all detected faces are anonymized). Your task is to **extend it with selective anonymization** - only anonymize registered family member faces, leaving strangers' faces untouched.

## 📂 Current System Architecture

### Existing Components

```
xla_demo/
├── models/
│   └── yolov11n-face.pt          # YOLOv11 face detection model
├── src/
│   ├── detector.py                # FaceDetector class (YOLOv11-based)
│   ├── privacy.py                 # Anonymization functions (blur, pixelate, etc.)
│   ├── face_lock.py               # Reversible encryption for faces (AES-GCM)
│   ├── bbox_smoother.py           # Temporal smoothing for bounding boxes
│   ├── anonymizer_backend.py      # FaceAnonymizationEngine with config presets
│   ├── backend_api.py             # FastAPI REST API
│   └── run_phone_cam.py           # Real-time camera demo script
├── test_face_lock.py              # Unit tests
├── benchmark_privacy_modes.py     # Performance benchmarking
└── evaluate_anonymization_security.py  # Security evaluation
```

### Key Technical Details

**1. Face Detection (`detector.py`):**

```python
class FaceDetector:
    def __init__(self, model_path="models/yolov11n-face.pt", conf_threshold=0.35)
    def detect(self, frame: np.ndarray) -> List[Tuple[List[int], float]]
    # Returns: [(box=[x,y,w,h], confidence), ...]
```

**2. Anonymization Modes (`privacy.py`):**

- `blur` - Gaussian blur with downscaling
- `pixelate` - Block pixelation
- `solid` - Solid color mask
- `neckup` - Anonymize head + neck region
- `headcloak` - Include hair, ears, jaw
- `silhouette` - Full head-and-shoulders (strongest)
- `obliterate` - Aggressive destruction (scrambling + color quantization)
- `lock` - Reversible encryption with AES-GCM (see `face_lock.py`)

**3. Face Lock System (`face_lock.py`):**

```python
class FaceRegionLocker:
    def __init__(self, passphrase: str)
    def lock_faces(self, frame, boxes, overlay_mode="rps") -> (frame, payloads)
    def unlock_faces(self, frame, encrypted_regions) -> frame
    # Uses AES-GCM encryption + Scrypt KDF
    # Supports overlay: solid, noise, ciphernoise, rps (reversible pixel shuffle)
```

**4. Real-time Demo (`run_phone_cam.py`):**

- Captures from camera (local camera, IP camera, or Camo virtual camera)
- Detects faces every N frames (default: every 2 frames)
- Applies anonymization mode
- Optional bbox smoothing (EMA or Kalman filter)
- Keyboard controls: Q=quit, M=switch mode, U=toggle unlock

### Current Configuration Presets

```python
# From anonymizer_backend.py
PRESETS = {
    "fast": Config(blur_scale=0.3, blur_kernel=21, pixel_block=20, ...),
    "balanced": Config(blur_scale=0.2, blur_kernel=31, pixel_block=16, ...),
    "strong": Config(blur_scale=0.15, blur_kernel=41, pixel_block=12, ...),
    "strict": Config(blur_scale=0.08, blur_kernel=51, pixel_block=8, ...)
}
```

## 🎯 New Feature Requirements

### Feature Goal

Implement **Selective Face Anonymization with Family Member Registration**:

1. **Registration Phase**: User registers family members by capturing their faces (5-10 photos from different angles)
2. **Storage**: Extract face embeddings and store in a local database (JSON file)
3. **Recognition Phase**: During real-time anonymization, detect faces → extract embeddings → match against database
4. **Selective Anonymization**:
   - **Mode 1 (Protect Family)**: Anonymize ONLY family members → strangers' faces remain visible
   - **Mode 2 (Show Family Only)**: Anonymize strangers → only family members remain visible

### Technical Requirements

#### 1. Face Recognition Module

**Create `src/face_recognition.py`:**

```python
class FaceRecognizer:
    """Extract face embeddings for recognition."""

    def __init__(self, model: str = "Facenet512"):
        # Use DeepFace library with Facenet512 model
        # Output: 512-dimensional embedding vector

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        # Extract embedding from face ROI
        # Input: Face image (BGR, any size)
        # Output: Vector [512] or None if extraction fails

    def extract_embeddings_from_frame(
        self,
        frame: np.ndarray,
        face_boxes: List[List[int]]
    ) -> List[Optional[np.ndarray]]:
        # Batch extraction for all detected faces

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Calculate similarity score (0-1, higher = more similar)

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Calculate distance (lower = more similar)
```

**Key Algorithms:**

- **Cosine Similarity**: `similarity = dot(v1, v2) / (norm(v1) * norm(v2))`
  - Range: 0-1 (typically 0.99+ for same person, <0.6 for different)
  - Threshold: 0.6-0.7 for matching
- **Euclidean Distance**: `distance = sqrt(sum((v1 - v2)^2))`
  - Range: 0-∞ (typically <0.6 for same person with Facenet512)
  - Threshold: 0.4-0.6 for matching

**Recommended Models:**

- **Facenet**: 128-d embeddings, fast, good accuracy
- **Facenet512**: 512-d embeddings, better accuracy, moderate speed (RECOMMENDED)
- **ArcFace**: 512-d, SOTA accuracy, moderate speed
- **VGG-Face**: 2622-d, high accuracy, slower

#### 2. Family Database Module

**Create `src/family_database.py`:**

```python
@dataclass
class FamilyMember:
    member_id: str                      # UUID
    name: str                           # Display name
    embeddings: List[List[float]]       # 5-10 embeddings from different angles
    registered_at: str                  # ISO timestamp
    metadata: dict                      # Optional: age, relation, etc.

class FamilyDatabase:
    """Manage family member face embeddings."""

    def __init__(self, db_path: str = "data/family_members.json"):
        # Load/save database from JSON file

    def add_member(
        self,
        name: str,
        embeddings: List[np.ndarray],
        metadata: dict = None
    ) -> str:
        # Add new family member, return member_id

    def remove_member(self, member_id: str) -> bool:
        # Remove family member from database

    def match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        metric: str = "cosine"
    ) -> Optional[Tuple[str, str, float]]:
        # Match against all stored embeddings
        # Returns: (member_id, name, score) or None
        # Algorithm: Compare with EACH stored embedding, return best match > threshold

    def get_all_members(self) -> List[FamilyMember]:
        # List all registered members
```

**Database Structure (JSON):**

```json
{
  "members": [
    {
      "member_id": "uuid-1234-5678",
      "name": "Nguyen Van A",
      "embeddings": [
        [0.234, -0.456, 0.789, ...],  // 512 values
        [0.231, -0.459, 0.791, ...],  // 512 values
        ...  // 5-10 embeddings
      ],
      "registered_at": "2026-03-13T10:30:00",
      "metadata": {"age": 25, "relation": "family"}
    }
  ]
}
```

#### 3. Selective Anonymization Module

**Create `src/selective_anonymizer.py`:**

```python
class SelectiveAnonymizer:
    """Anonymize only registered family member faces or strangers."""

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        database: FamilyDatabase,
        match_threshold: float = 0.6,
        anonymize_mode: str = "blur"
    ):
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        anonymize_target: str = "family"  # "family" or "stranger"
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process frame with selective anonymization.

        Workflow:
        1. Detect all faces using self.detector.detect(frame)
        2. Extract embeddings for each detected face
        3. Match each embedding against self.database.match()
        4. Determine faces to anonymize:
           - If anonymize_target="family": anonymize matched faces
           - If anonymize_target="stranger": anonymize non-matched faces
        5. Apply anonymization using privacy.anonymize_faces()

        Returns:
            (anonymized_frame, detection_info)

            detection_info structure:
            [
                {
                    "box": [x, y, w, h],
                    "confidence": 0.95,
                    "matched": True,
                    "member_name": "Nguyen Van A",
                    "match_score": 0.87,
                    "anonymized": True
                },
                ...
            ]
        """
        pass
```

**Algorithm Flow:**

```
Input: frame
  ↓
┌─────────────────────────────────────┐
│ 1. Detect Faces                     │
│    detections = detector.detect()   │
│    → [(box, conf), (box, conf), ...]│
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 2. Extract Embeddings               │
│    for each face:                   │
│      face_roi = frame[y:y+h, x:x+w] │
│      emb = recognizer.extract()     │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. Match Against Database           │
│    for each embedding:              │
│      match = database.match(emb)    │
│      if match:                      │
│        is_family = True             │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 4. Select Faces to Anonymize        │
│    if anonymize_target == "family": │
│      boxes_to_blur = [family faces] │
│    else:                            │
│      boxes_to_blur = [stranger faces]│
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 5. Apply Anonymization              │
│    result = anonymize_faces(        │
│      frame,                         │
│      boxes_to_blur,                 │
│      mode=anonymize_mode            │
│    )                                │
└─────────────────────────────────────┘
  ↓
Output: anonymized_frame, detection_info
```

#### 4. Registration Tool

**Create `src/registration_tool.py`:**

```python
class RegistrationTool:
    """Interactive tool for registering family members."""

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        database: FamilyDatabase
    ):
        pass

    def register_from_camera(
        self,
        camera_source: int = 0,
        name: str = None,
        num_captures: int = 10
    ) -> bool:
        """
        Register family member from camera.

        Interactive flow:
        1. Open camera window
        2. Show live preview with face detection boxes
        3. Instructions: "Press SPACE to capture, Q to quit"
        4. Status overlay: "Captured: 3/10"
        5. Validation: Only accept frames with EXACTLY 1 face detected
        6. For each capture:
           - Detect face
           - Extract face ROI
           - Extract embedding
           - Store embedding
        7. After capturing all embeddings:
           - Save to database.add_member(name, embeddings)
           - Display success message

        UI Requirements:
        - Draw green box if 1 face detected
        - Draw red box if 0 or >1 faces detected
        - Show capture count: "Captured: X/10"
        - Show instructions on screen

        Returns: True if registration successful
        """
        pass

    def list_members(self) -> None:
        """Display registered family members."""
        # Print formatted list with member_id, name, num_embeddings, registered_at
```

**Command-line Interface:**

```bash
# Register new family member
python registration_tool.py --register --name "Nguyen Van A" --captures 10

# List all registered members
python registration_tool.py --list

# Remove member
python registration_tool.py --remove --member-id "uuid-1234"
```

#### 5. Integration with run_phone_cam.py

**Add new command-line arguments:**

```python
parser.add_argument(
    "--selective-mode",
    type=str,
    choices=["disabled", "family", "stranger"],
    default="disabled",
    help="Selective anonymization mode: disabled, family (anonymize family members), or stranger (anonymize strangers)"
)

parser.add_argument(
    "--face-db",
    type=str,
    default="data/family_members.json",
    help="Path to family member database"
)

parser.add_argument(
    "--recognition-threshold",
    type=float,
    default=0.6,
    help="Similarity threshold for face matching (0-1, higher=stricter)"
)

parser.add_argument(
    "--recognition-model",
    type=str,
    default="Facenet512",
    choices=["Facenet", "Facenet512", "ArcFace", "VGG-Face"],
    help="Face recognition model"
)
```

**Modified main loop:**

```python
# Initialize selective anonymization if enabled
selective_anonymizer = None
if args.selective_mode != "disabled":
    recognizer = FaceRecognizer(model=args.recognition_model)
    database = FamilyDatabase(db_path=args.face_db)
    selective_anonymizer = SelectiveAnonymizer(
        detector=detector,
        recognizer=recognizer,
        database=database,
        match_threshold=args.recognition_threshold,
        anonymize_mode=args.anonymize_mode
    )

# In main loop:
if selective_anonymizer:
    frame, detection_info = selective_anonymizer.process_frame(
        frame,
        anonymize_target=args.selective_mode
    )

    # Optional: Draw labels showing matched/stranger status
    for info in detection_info:
        if info["matched"]:
            label = f"Family: {info['member_name']} ({info['match_score']:.2f})"
            color = (0, 255, 0)  # Green for family
        else:
            label = "Stranger"
            color = (0, 0, 255)  # Red for stranger

        x, y, w, h = info["box"]
        cv2.putText(frame, label, (x, y-10), ...)
else:
    # Original universal anonymization
    frame = anonymize_faces(frame, boxes, mode=args.anonymize_mode)
```

## 📦 Dependencies

**Add to `requirements.txt`:**

```
deepface>=0.0.79
tf-keras>=2.17.0
# Or alternatively:
# face-recognition>=1.3.0  # dlib-based alternative
```

**Installation:**

```bash
pip install deepface tf-keras
```

**Note:** DeepFace will auto-download model weights on first run (~100MB for Facenet512).

## 🎯 Implementation Strategy

### Phase 1: Core Recognition System (Priority: HIGH)

1. ✅ Implement `face_recognition.py`
   - FaceRecognizer class with DeepFace
   - Test with sample images
   - Verify embeddings are consistent

2. ✅ Implement `family_database.py`
   - JSON-based storage
   - CRUD operations (add, remove, list)
   - Matching with cosine similarity

3. ✅ Test integration
   - Create test script: register 2-3 people, test matching
   - Verify false positive rate < 5%
   - Verify false negative rate < 5%

### Phase 2: Selective Anonymization (Priority: HIGH)

4. ✅ Implement `selective_anonymizer.py`
   - Integrate detector + recognizer + database
   - Implement selective logic
   - Return detection metadata

5. ✅ Test with static images
   - Prepare test images with known/unknown faces
   - Verify correct faces are anonymized

### Phase 3: Registration Tool (Priority: MEDIUM)

6. ✅ Implement `registration_tool.py`
   - Interactive camera capture
   - UI with OpenCV
   - Save to database

7. ✅ Test registration workflow
   - Register 3-5 family members
   - Test from different angles/lighting

### Phase 4: Integration (Priority: HIGH)

8. ✅ Modify `run_phone_cam.py`
   - Add command-line arguments
   - Integrate SelectiveAnonymizer
   - Add visual feedback (labels)

9. ✅ End-to-end testing
   - Register family members
   - Run real-time demo
   - Test both modes (family/stranger)

### Phase 5: Optimization (Priority: LOW)

10. ⚡ Performance optimization
    - Cache embeddings for tracked faces
    - Extract embeddings every N frames (not every frame)
    - Use threading for embedding extraction
11. ⚡ Advanced features (optional)
    - Multiple face databases (work, family, friends)
    - Confidence-based blur intensity
    - Face tracking for smooth transitions

## 🔍 Testing Criteria

### Unit Tests

```python
# test_face_recognition.py
def test_embedding_extraction():
    # Test embedding extraction returns 512-d vector
    # Test consistency across multiple calls

def test_cosine_similarity():
    # Test same person: similarity > 0.8
    # Test different person: similarity < 0.5

# test_family_database.py
def test_add_remove_member():
    # Test CRUD operations

def test_matching():
    # Test matching returns correct member
    # Test threshold behavior

# test_selective_anonymizer.py
def test_selective_anonymization():
    # Test family mode anonymizes only family
    # Test stranger mode anonymizes only strangers
```

### Integration Tests

```bash
# Test 1: Register member
python registration_tool.py --register --name "Test Person"

# Test 2: List members
python registration_tool.py --list

# Test 3: Run selective anonymization
python src/run_phone_cam.py --selective-mode family --show --debug-draw

# Test 4: Verify correct faces are anonymized
# Expected: Only registered faces are blurred
```

### Performance Benchmarks

**Target metrics:**

- Registration: < 10 seconds for 10 captures
- Face recognition: < 100ms per face
- Total FPS: > 10 FPS with 2-3 faces in frame
- False positive rate: < 5%
- False negative rate: < 5%

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: False Positives (Strangers recognized as family)

**Solution:**

- Increase `match_threshold` from 0.6 to 0.7
- Capture more embeddings during registration (15-20 instead of 10)
- Use stricter model (ArcFace instead of Facenet)

### Pitfall 2: False Negatives (Family not recognized)

**Solution:**

- Decrease `match_threshold` from 0.6 to 0.5
- Register with diverse conditions (angles, lighting, expressions)
- Check embedding quality during registration

### Pitfall 3: Low FPS

**Solution:**

- Extract embeddings every 5-10 frames, use tracking in between
- Use faster model (Facenet instead of Facenet512)
- Implement embedding caching for tracked faces
- Run embedding extraction in separate thread

### Pitfall 4: Poor Lighting Recognition

**Solution:**

- Register with multiple lighting conditions
- Use data augmentation during registration (brightness variations)
- Add preprocessing: histogram equalization, gamma correction

## 📊 Expected Results

### Registration Phase

```
📸 Registering: Nguyen Van A
✓ Captured 1/10
✓ Captured 2/10
...
✓ Captured 10/10
✅ Registration successful!
   Name: Nguyen Van A
   ID: 7d4f23a1-...
   Embeddings: 10
```

### Selective Anonymization (Real-time)

```
Frame: 120 | FPS: 15.3
Faces detected: 3
  [1] Family: Nguyen Van A (0.87) → ANONYMIZED
  [2] Family: Tran Thi B (0.92) → ANONYMIZED
  [3] Stranger (no match) → VISIBLE
```

## 🎓 Learning Resources

- **DeepFace**: https://github.com/serengil/deepface
- **FaceNet Paper**: https://arxiv.org/abs/1503.03832
- **ArcFace Paper**: https://arxiv.org/abs/1801.07698
- **Face Recognition Tutorial**: https://www.pyimagesearch.com/face-recognition/

---

## ✅ Definition of Done

This feature is complete when:

1. ✅ User can register family members via registration tool
2. ✅ Database correctly stores and retrieves embeddings
3. ✅ Face matching achieves >95% accuracy on test set
4. ✅ Selective anonymization works in both modes (family/stranger)
5. ✅ Real-time demo runs at >10 FPS with 2-3 faces
6. ✅ Command-line interface is intuitive and documented
7. ✅ Code is tested with unit tests
8. ✅ Documentation is complete (README, docstrings)

---

**Implementation Time Estimate:** 4-6 hours for experienced developer

**Key Technologies:** Python, OpenCV, DeepFace, NumPy, JSON

**Difficulty Level:** Intermediate (requires understanding of embeddings, similarity metrics, and real-time video processing)
