# AI Prompt Log — deep26 Week 4

## 1. 모델 선택 및 LoRA 설정
**Prompt:** Vision-Language Navigation을 위한 action classification에 적합한 VLM과 fine-tuning 방법을 추천해달라.
**AI 활용:** Qwen3-VL-2B-Instruct + LoRA (r=16, alpha=32, target: q_proj, v_proj) 구성 채택

## 2. Collator 레이블 정렬 버그 수정
**Prompt:** 학습 loss가 감소하지 않고 invalid output rate가 1.0으로 고정된다. collator의 label masking이 잘못된 것 같다.
**AI 활용:** image token expansion으로 인해 standalone tokenization 사용 불가 확인. input_ids에서 직접 assistant block 위치를 탐색하는 방식으로 수정.

## 3. 에피소드 단위 데이터 분할
**Prompt:** train/val split을 어떻게 해야 하는가? frame 단위로 나눠도 되는가?
**AI 활용:** frame-level split이 금지된 이유 이해 및 episode-level split 구현

## 4. W&B 통합
**Prompt:** 학습 중 loss, accuracy, learning rate를 W&B로 기록하는 방법을 알려달라.
**AI 활용:** wandb.init(), wandb.log() 구현, per-step 및 per-epoch 메트릭 로깅

## 5. GPU 병목 현상 진단
**Prompt:** nvidia-smi에서 GPU utilization이 0%인데 VRAM은 꽉 차있다. 원인과 해결방법은?
**AI 활용:** collator가 main process에서 실행되는 구조적 문제 진단. num_workers 증가, worker_init_fn 활용 등 여러 방법 시도

## 6. VRAM OOM 크래시 해결
**Prompt:** epoch 2에서 step 600-800 구간에 반복적으로 OOM이 발생한다.
**AI 활용:** expandable_segments, empty_cache, try/except OOM handler 적용

## 7. 체크포인트 재개 문제
**Prompt:** checkpoint에서 resume할 때 epoch 번호가 잘못 계산된다.
**AI 활용:** 0-indexed epoch 저장 방식 파악, checkpoint_latest.pt vs mid-epoch checkpoint 구분

## 8. 클로즈드-루프 평가 스크립트
**Prompt:** IsaacSim에서 모델을 실행하고 SR/OSR/SPL/nDTW/Goal Distance를 계산하는 평가 스크립트를 작성해달라.
**AI 활용:** week4_closed_loop_eval.py 전체 구현
