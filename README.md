# MultiHopUSRP

실험 환경의 주요 스크립트를 빠르게 실행할 수 있도록 정리한 가이드입니다. 기본적으로 5G NR long PRACH 파형을 생성한 뒤 송/수신하고, 패킷 캡처 및 지연 분석까지 이어지는 워크플로우를 다룹니다.

---

## 1. 개발 환경 준비

```bash
conda env create -f env/environment.yml
conda activate usrp
```

USRP IP/직렬 번호, 관리용 주소 등은 `scripts/tx_flowgraph.py`, `scripts/rx_flowgraph.py` 실행 시 옵션으로 덮어씁니다. (예시의 기본값은 N320 장비에 맞춰져 있음)

---

## 2. 파형 생성 (`scripts/waveform_generator.py`)

기본 설정은 5G NR long PRACH (N_ZC=839, M=64, CP=3168)입니다.

```bash
python scripts/waveform_generator.py \
  --waveform nr_prach_long \
  --packet-len 839 \
  --zc-root 1 \
  --output outputs/prach_long.npy \
  --plot outputs/prach_long.png
```

- `--nr-prach-m-count`, `--nr-prach-m-index`, `--nr-prach-cp-len`, `--nr-prach-ncs`로 PRACH 파라미터 조정 가능  
- 플롯은 앞 1000샘플에 대해 Real/Imag/Amplitude/Phase(unwrap) 4개 축을 표시  
- 다른 유형(Zadoff-Chu 단일패킷, 모노톤 등)이 필요하면 `--waveform`을 변경

---

## 3. 송신 (`scripts/tx_flowgraph.py`)

```bash
python scripts/tx_flowgraph.py \
  --waveform-file outputs/prach_long.npy \
  --addr addr=192.168.10.1 \
  --mgmt-addr mgmt_addr=192.168.100.8 \
  --freq 2.45e9 \
  --samp-rate 1e6 \
  --gain 10
```

- `--waveform-file` 는 반드시 `.npy` 복소64 샘플 파일  
- `--repeat` 를 주면 무한 반복 송신  
- `--tx-antenna`, `--clock-source`, `--time-source`, `--lo-offset` 등 UHD 옵션 사용 가능  
- 송신이 끝나면 인터페이스 패킷 카운터(`tx_packets`) 요약 출력

---

## 4. 수신 (`scripts/rx_flowgraph.py`)

```bash
python scripts/rx_flowgraph.py \
  --output captures/prach_rx.c32 \
  --addr addr=192.168.10.2 \
  --mgmt-addr mgmt_addr=192.168.100.7 \
  --freq 2.45e9 \
  --samp-rate 1e6 \
  --gain 20 \
  --duration 10
```

- 결과 파일은 복소 32비트 raw (`.c32`)  
- `--duration 0` 으로 두면 Ctrl-C 까지 계속 녹음  
- `--sync-start` 로 USRP 시간을 0으로 초기화 가능  
- 디렉터리가 없으면 자동 생성

---

## 5. 지연 분석 (`scripts/analyze_delay.py`)

```bash
python scripts/analyze_delay.py \
  --tx-waveform outputs/prach_long.npy \
  --rx-capture captures/prach_rx.c32 \
  --samp-rate 1e6 \
  --plot outputs/corr.png \
  --waveform-plot outputs/rx_wave.png \
  --print-rx-samples 8
```

- FFT 기반 상관으로 `sample_delay`, `time_delay`, 정규화 피크 출력  
- `--plot` 은 상관 계수 크기를 그림으로 저장  
- `--waveform-plot` 은 수신 파형 Real/Imag/|x|/Phase 플롯  
- `--allow-negative`, `--max-lag` 로 검색 범위 제한 가능

---

## 6. 캡처 파이프라인 (`scripts/run_capture_pipeline.py`)

송신/수신 중 NIC 트래픽을 `dumpcap`으로 자동 캡처합니다. `dumpcap` 권한은 미리 capability 설정(`setcap cap_net_raw,cap_net_admin=eip $(which dumpcap)`)을 권장합니다.

### 송신 패킷 캡처
```bash
python scripts/run_capture_pipeline.py tx \
  --interface ens4f0np0 \
  --pcap outputs/tx_capture.pcap \
  --udp-port 49152 \
  --timestamp-mode adapter_unsynced \
  --tx-args --waveform-file outputs/prach_long.npy --gain 10 --freq 2.45e9
```

### 수신 패킷 캡처
```bash
python scripts/run_capture_pipeline.py rx \
  --interface ens4f1np1 \
  --pcap outputs/rx_capture.pcap \
  --udp-port 49153 \
  --timestamp-mode adapter_unsynced \
  --rx-args --duration 5 --output captures/prach_rx.c32
```

- `--timestamp-mode` 는 `dumpcap --list-time-stamp-types` 결과에 맞게 지정  
- `--extra-dumpcap-flags`, `--ring-buffer-mb`, `--use-sudo` 등 추가 옵션 제공  
- Flowgraph가 종료되면 `dumpcap`을 SIGINT로 정리하며 `.pcap` 파일 저장

**주의**: 송신 시 USRP `tx_packets` 증가로 패킷 수 확인 가능. 시스템에서 `net.core.wmem_max` 경고가 나오면 권장 값으로 조정 필요.

---

## 7. VITA-49 / 패킷 후처리

`outputs/tx_capture.pcap`, `outputs/rx_capture.pcap`에는 UHD가 전송한 VITA-49 UDP 프레임이 기록됩니다. Scapy, Pyshark 등을 이용해 시퀀스/타임스탬프를 파싱한 뒤 `scripts/analyze_delay.py`의 `sample_delay`와 결합하면, 특정 샘플이 실린 패킷 번호 및 PHC 시각을 매핑할 수 있습니다. (향후 자동화 스크립트는 `scripts/utils/` 참고)

---

## 8. 유틸리티 (`scripts/utils`)

- `terminate_uhd_claim.py`: 이미 claim된 USRP 세션 강제 종료  
- `packet_monitor.py`: NIC 통계를 감시하거나 캡처 중 드롭 확인  
- README는 `scripts/utils/README.md` 참고

---

## 9. 권장 워크플로우 요약

1. PRACH 파형 생성 (`waveform_generator.py`)  
2. 송신/수신 플로우그래프로 연결 확인  
3. `run_capture_pipeline.py`로 패킷 + 센서 데이터 동시 기록  
4. `analyze_delay.py`로 샘플 지연 계산  
5. 필요시 pcap 파싱으로 하드웨어 타임스탬프와 비교  

실험 전에는 USRP 장치가 다른 프로세스에 `claimed`되어 있지 않은지 확인하고(`scripts/utils/terminate_uhd_claim.py`) PTP/PHC 동기화(`ptp4l`, `phc2sys`)를 맞춰두면 하드웨어 타임스탬프 비교가 용이합니다.

---

시스템 구성/파라미터가 실험 환경마다 다르므로, 위 명령에 등장하는 IP와 장치명은 반드시 실제 장비에 맞게 수정하세요. 모든 스크립트는 `--help`를 통해 추가 옵션과 사용법을 확인할 수 있습니다.

