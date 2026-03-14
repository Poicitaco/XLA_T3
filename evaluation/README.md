# 📊 Evaluation & Benchmarking

Thư mục này chứa các scripts và kết quả đánh giá hiệu suất, bảo mật của hệ thống Face Anonymization.

## 📁 Files

### Benchmark Scripts

#### [benchmark_privacy_modes.py](benchmark_privacy_modes.py)

**Đo lường hiệu suất của các chế độ anonymization**

**Metrics:**

- FPS (Frames Per Second)
- Latency (milliseconds per frame)
- SSIM (Structural Similarity Index) - so sánh chất lượng hình ảnh

**Modes được test:**

- `blur`, `pixelate`, `solid`, `neckup`, `headcloak`, `silhouette`

**Presets:**

- `fast`, `balanced`, `strong`, `strict`

**Usage:**

```bash
# Default: 100 frames, balanced preset
python evaluation/benchmark_privacy_modes.py

# Custom: 200 frames, all presets
python evaluation/benchmark_privacy_modes.py --frames 200 --preset all

# From video file
python evaluation/benchmark_privacy_modes.py --video input.mp4 --frames 500
```

**Output:** `benchmark_results.csv`

---

#### [evaluate_anonymization_security.py](evaluate_anonymization_security.py)

**Đánh giá độ bảo mật của anonymization (face recognition resistance)**

**Metrics:**

- Cosine Similarity giữa original và anonymized embeddings
- Euclidean Distance
- Face detection success rate sau khi anonymize

**Workflow:**

1. Capture frames có khuôn mặt
2. Extract face embeddings (original)
3. Apply anonymization
4. Extract embeddings (anonymized)
5. Calculate similarity/distance

**Usage:**

```bash
# Default: 50 frames, balanced preset
python evaluation/evaluate_anonymization_security.py

# Custom: 100 frames, all modes
python evaluation/evaluate_anonymization_security.py --frames 100 --mode all

# Specific preset
python evaluation/evaluate_anonymization_security.py --preset strong
```

**Output:** `security_evaluation.csv`

---

#### [security_evaluation.py](security_evaluation.py)

**Legacy script cho security evaluation (older version)**

---

#### [benchmark.py](benchmark.py)

**Simple benchmark script (older version)**

---

### Result Files

#### [benchmark_results.csv](benchmark_results.csv)

**Kết quả benchmark hiệu suất**

**Columns:**

- `mode` - Anonymization mode
- `preset` - Configuration preset
- `fps` - Frames per second
- `latency_ms` - Average latency in milliseconds
- `ssim` - Structural Similarity Index (0-1, càng cao càng giống original)

**Sample data:**

```csv
mode,preset,fps,latency_ms,ssim
blur,balanced,21.53,46.45,0.8934
pixelate,balanced,33.95,29.45,0.7821
silhouette,balanced,8.72,114.70,0.6311
```

---

#### [security_evaluation.csv](security_evaluation.csv)

**Kết quả đánh giá bảo mật**

**Columns:**

- `mode` - Anonymization mode
- `preset` - Configuration preset
- `avg_cosine_similarity` - Average similarity (0-1, càng thấp càng bảo mật)
- `avg_euclidean_distance` - Average distance (càng cao càng bảo mật)
- `min_similarity` - Minimum similarity observed
- `max_similarity` - Maximum similarity observed

**Sample data:**

```csv
mode,preset,avg_cosine_similarity,avg_euclidean_distance,min_similarity,max_similarity
blur,balanced,0.234,1.456,0.123,0.345
silhouette,strong,0.089,2.234,0.012,0.178
```

---

## 🎯 Key Findings

### Performance (từ benchmark_results.csv)

**Fastest modes (by FPS):**

1. `pixelate` - 33.95 FPS
2. `blur` - 21.53 FPS
3. `solid` - 20+ FPS

**Slowest modes:**

1. `silhouette` - 8.72 FPS (most computation)
2. `obliterate` - ~10 FPS

**Image Quality (SSIM):**

- `solid` - 1.0 (no degradation, just masking)
- `blur` - 0.89 (good visual quality)
- `silhouette` - 0.63 (significant alteration)

### Security (từ security_evaluation.csv)

**Most secure (lowest similarity = harder to recognize):**

1. `silhouette` - avg similarity: 0.089
2. `obliterate` - avg similarity: 0.12
3. `headcloak` - avg similarity: 0.18

**Least secure:**

1. `blur` (fast preset) - avg similarity: 0.45
2. `pixelate` (fast preset) - avg similarity: 0.38

**Recommendation:**

- **For real-time with good privacy:** `headcloak` preset `balanced` (FPS ~15, similarity ~0.18)
- **For maximum privacy:** `silhouette` preset `strong` (FPS ~8, similarity ~0.089)
- **For performance:** `pixelate` preset `fast` (FPS ~33, similarity ~0.38)

---

## 🚀 Running Evaluations

### Complete Evaluation Suite

```bash
# 1. Benchmark performance (all modes, all presets)
python evaluation/benchmark_privacy_modes.py --frames 200 --preset all

# 2. Security evaluation (all modes)
python evaluation/evaluate_anonymization_security.py --frames 100 --mode all

# 3. Analyze results
# Open benchmark_results.csv and security_evaluation.csv in Excel/Python
```

### Visualization (Optional)

```python
# Create charts from results
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_perf = pd.read_csv('evaluation/benchmark_results.csv')
df_sec = pd.read_csv('evaluation/security_evaluation.csv')

# Plot FPS comparison
df_perf.plot(x='mode', y='fps', kind='bar', title='FPS by Mode')
plt.savefig('fps_comparison.png')

# Plot security vs performance
plt.scatter(df_sec['avg_cosine_similarity'], df_perf['fps'])
plt.xlabel('Cosine Similarity (lower = more secure)')
plt.ylabel('FPS (higher = faster)')
plt.title('Privacy vs Performance Trade-off')
plt.savefig('privacy_vs_performance.png')
```

---

## 📝 Adding New Evaluations

### Template for New Benchmark

```python
"""
New evaluation: [Description]
"""
import argparse
import cv2
import pandas as pd
from src.detector import FaceDetector
from src.privacy import anonymize_faces

def evaluate():
    # 1. Setup
    detector = FaceDetector()

    # 2. Capture/load test data
    # ...

    # 3. Run evaluation
    results = []
    for frame in test_frames:
        # Detect, anonymize, measure
        result = {
            'metric1': value1,
            'metric2': value2,
        }
        results.append(result)

    # 4. Save results
    df = pd.DataFrame(results)
    df.to_csv('evaluation/new_evaluation_results.csv', index=False)

if __name__ == "__main__":
    evaluate()
```

---

## 📚 Related Documentation

- [../docs/IMPLEMENTATION_PROMPT.md](../docs/IMPLEMENTATION_PROMPT.md) - Technical implementation details
- [../README.md](../README.md) - Project overview
- [../src/](../src/) - Source code

---

**Last Updated:** 2026-03-13  
**Performance Target:** >10 FPS với 2-3 faces, similarity <0.3 cho modes bảo mật cao
