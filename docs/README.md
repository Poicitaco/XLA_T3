# 📚 Documentation

Thư mục này chứa tài liệu kỹ thuật cho dự án Face Anonymization System.

## Files

### [IMPLEMENTATION_PROMPT.md](IMPLEMENTATION_PROMPT.md)

**Prompt chi tiết đầy đủ cho việc implement tính năng Selective Face Anonymization**

Bao gồm:

- Kiến trúc hệ thống hiện tại
- Yêu cầu kỹ thuật chi tiết
- Algorithm flow với diagrams
- Implementation strategy (4 phases)
- Testing criteria & benchmarks
- Common pitfalls & solutions
- Performance optimization tips

**Đối tượng:** Developer cần hiểu toàn bộ hệ thống

---

### [AI_GENERATION_PROMPT.md](AI_GENERATION_PROMPT.md)

**Prompt ngắn gọn để generate code với AI (ChatGPT/Claude)**

Bao gồm:

- Quick context (3 dòng)
- Code structure cho 4 modules cần implement
- Existing code để integrate
- Technical requirements
- Sample output format
- Testing checklist

**Đối tượng:** Copy-paste vào ChatGPT/Claude để generate code

---

## Tính năng: Selective Face Anonymization

### Mục đích

Cho phép đăng ký người thân và chỉ che mặt họ (hoặc ngược lại - chỉ che người lạ).

### Architecture Overview

```
Registration Phase:
  Camera → Capture 10 photos → Extract embeddings → Save to JSON

Recognition Phase:
  Frame → Detect faces → Extract embeddings → Compare with DB →
  Match (similarity > 0.6) → Selective anonymization

Anonymization Phase:
  Mode "family": Blur matched faces, keep strangers visible
  Mode "stranger": Blur non-matched faces, keep family visible
```

### Tech Stack

- **Face Recognition**: DeepFace with Facenet512 model
- **Embeddings**: 512-dimensional vectors
- **Similarity Metric**: Cosine similarity (threshold: 0.6)
- **Database**: JSON file at `data/family_members.json`

### Modules to Implement

1. `src/face_recognition.py` - Extract face embeddings
2. `src/family_database.py` - JSON database with CRUD operations
3. `src/selective_anonymizer.py` - Selective anonymization logic
4. `registration_tool.py` - Interactive registration tool

---

## Usage

### For Developers

1. Read [IMPLEMENTATION_PROMPT.md](IMPLEMENTATION_PROMPT.md) để hiểu toàn bộ system
2. Follow implementation strategy từng phase
3. Test theo testing criteria được define

### For AI-Assisted Development

1. Copy nội dung [AI_GENERATION_PROMPT.md](AI_GENERATION_PROMPT.md)
2. Paste vào ChatGPT/Claude
3. Review và integrate generated code

---

## Related Documents

- [../README.md](../README.md) - Project overview và usage instructions
- [../evaluation/README.md](../evaluation/README.md) - Benchmark và security evaluation results
