## Raspberry Pi 5 + Hailo AT 기반 실시간 비전 캡셔닝

이 예제는 Raspberry Pi 5에서 **USB 카메라 → Hailo AT NPU 추론 → LLM 캡션 생성 → TTS 음성 출력** 파이프라인을 구현한 Python 3 스크립트입니다. Hailo 가속기가 없는 개발 환경에서도 `--demo` 플래그를 사용하면 목업 감지를 이용해 빠르게 동작 확인이 가능합니다.

### 요구 사항
- Python 3.10+
- Raspberry Pi 5, USB 카메라
- Hailo-8 기반 Hailo AT NPU 모듈 + HEF 모델 파일 (예: `yolov5s.hef`)
- OpenAI API 키 (모델: `gpt-4o-mini` 등 비전 입력 지원 모델)
- ALSA 기반 오디오 출력(헤드폰/스피커)

### 필수 패키지 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r vision_captioning/requirements.txt

# Hailo SDK/드라이버는 Hailo 공식 설치 스크립트로 별도 설치
```

### 실행 예시
```bash
export OPENAI_API_KEY=sk-...
python -m vision_captioning.app \
  --hef-path /path/to/yolov5s.hef \
  --prompt "You are a concise Korean captioning agent. Describe the scene clearly." \
  --camera-index 0 \
  --speak
```

Hailo 장치를 연결하지 않았거나 빠른 기능 점검이 필요한 경우:
```bash
python -m vision_captioning.app --demo --speak
```

### 주요 CLI 옵션
- `--demo`: Hailo 없이 목업 감지기로 빠르게 실행.
- `--no-llm`: LLM 호출을 생략하고 감지 결과만 로그/음성으로 출력.
- `--interval`: 캡션 주기(초). 기본 2초.
- `--frame-width/--frame-height`: 입력 프레임 크기 조정.
- `--speak`: 캡션을 음성으로 합성.

### 파일 구성
- `app.py`: CLI 엔트리포인트.
- `pipeline.py`: 카메라 캡처 → 감지 → 캡션 → 음성 합성 파이프라인 로직.
- `config.py`: 환경 및 기본 파라미터 관리.
- `detector.py`: Hailo 감지기 및 목업 감지기 구현.
- `llm_client.py`: OpenAI Vision LLM을 이용한 캡션 생성.
- `tts_client.py`: `pyttsx3` 기반 로컬 TTS 출력.

### Raspberry Pi 5 최적화 팁
- `--frame-width/--frame-height`로 입력 해상도를 줄여 LLM 인코딩 비용 절감.
- `--interval`을 이용해 프레임 캡처 주기를 조절(기본 2초).
- Hailo HEF와 YAML 후처리 설정 파일을 SD 카드 대신 NVMe/USB SSD에 배치하여 로드 속도 개선.
- 음성 재생 지연을 줄이기 위해 `pyttsx3`의 드라이버는 기본 `espeak`를 사용합니다.

### 보안 고려 사항
- API 키는 `.env` 파일에 보관하거나 `OPENAI_API_KEY` 환경 변수로 주입하세요.
- 네트워크가 차단된 환경에서는 `--demo` 모드로 LLM 호출을 건너뛰고 감지 결과만 로그로 확인할 수 있습니다.
